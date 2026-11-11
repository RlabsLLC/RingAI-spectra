import os
import time
import threading
import queue
import io
import zipfile
import math
import wave
import struct
from flask import Flask, request, jsonify, render_template_string, send_file

# ---------------------------------------------------------------------------
# Global State & Queues
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Producer-Consumer queues
# stream_queue holds URLs to be downloaded.
# training_queue holds paths to downloaded audio files ready for training.
stream_queue = queue.Queue()
training_queue = queue.Queue()

# Global state dictionary for the web UI to poll
state = {
    "project_name": "My Noise Engine",
    "model_architecture": "1B Dense",
    "moe_activation_params": False,
    "max_song_longevity_sec": 300,
    "hf_repo_id": "",
    "hf_token": "",
    "is_training": False,
    "download_active": True,
    "logs": ["System initialized. Waiting for commands..."],
    "files_ready": 0,
    "epochs_completed": 0
}

def log_msg(msg):
    """Helper to add messages to the global log (keeps last 50)"""
    timestamp = time.strftime("%H:%M:%S")
    state["logs"].insert(0, f"[{timestamp}] {msg}")
    if len(state["logs"]) > 50:
        state["logs"].pop()

# ---------------------------------------------------------------------------
# Background Workers (The "Engine")
# ---------------------------------------------------------------------------

def audio_downloader_worker():
    """
    Thread 1: The Producer.
    Continuously checks for URLs to download. Uses yt-dlp to extract audio streams.
    Passes downloaded/streamed file paths to the training queue.
    """
    log_msg("Audio Downloader thread started.")
    
    # Setup yt-dlp options for best audio extraction
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloads/%(id)s.%(ext)s',
        'noplaylist': False, # ALLOW PLAYLISTS
        'quiet': True,
        'extract_audio': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    # Ensure download directory exists
    if not os.path.exists('downloads'):
        os.makedirs('downloads')

    while True:
        try:
            # Block until a URL is added to the queue
            url = stream_queue.get(timeout=2)
            log_msg(f"Fetching audio stream from: {url}")
            
            try:
                import yt_dlp
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # extract_info with download=False first to get playlist metadata fast
                    info = ydl.extract_info(url, download=False)
                    entries = info.get('entries', [info]) # Handle both single videos and playlists
                    
                    for entry in entries:
                        if not entry: continue
                        log_msg(f"Downloading track: {entry.get('title', 'Unknown')}")
                        
                        # Download track individually so training can start immediately
                        ydl.download([entry['webpage_url']])
                        
                        # Calculate expected output filename
                        expected_filename = f"downloads/{entry['id']}.wav"
                        
                        # Send to the training thread
                        training_queue.put(expected_filename)
                        state["files_ready"] = training_queue.qsize()
                        log_msg(f"Successfully buffered track: {expected_filename}")
                
            except Exception as e:
                log_msg(f"Download failed (Is FFmpeg installed?): {str(e)}")
            
            stream_queue.task_done()
            
        except queue.Empty:
            continue

def model_training_worker():
    """
    Thread 2: The Consumer.
    Waits for audio files to appear in the training queue.
    If training is active, it processes them step-by-step.
    """
    log_msg("Training Engine thread started.")
    while True:
        if not state["is_training"]:
            time.sleep(1)
            continue
            
        try:
            # Try to get an audio file, wait up to 2 seconds
            audio_file = training_queue.get(timeout=2)
            state["files_ready"] = training_queue.qsize()
            
            log_msg(f"Training step started on: {audio_file}")
            log_msg(f"Using {state['model_architecture']} with MoE active: {state['moe_activation_params']}")
            
            # Simulate PyTorch forward/backward pass time
            time.sleep(4) 
            
            # Simulate applying the longevity limit
            log_msg(f"Processed up to max longevity of {state['max_song_longevity_sec']}s")
            
            # Cleanup simulated file
            if os.path.exists(audio_file):
                os.remove(audio_file)
                
            log_msg(f"Finished training step. Audio data flushed.")
            state["epochs_completed"] += 1
            training_queue.task_done()
            
        except queue.Empty:
            # This allows the "start with 10, let the rest load later" feature.
            # If the queue is empty, it just waits patiently for the downloader.
            log_msg("Training queue empty. Waiting for Downloader thread to fetch more audio...")
            time.sleep(3)

# Start background threads
threading.Thread(target=audio_downloader_worker, daemon=True).start()
threading.Thread(target=model_training_worker, daemon=True).start()


# ---------------------------------------------------------------------------
# Web UI (HTML/JS)
# ---------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Noise Engine Controller</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .tab-btn.active { border-bottom: 2px solid #3b82f6; color: #3b82f6; }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen font-sans">
    
    <div class="max-w-5xl mx-auto p-6">
        <header class="mb-8 flex justify-between items-end border-b border-gray-700 pb-4">
            <div>
                <h1 class="text-3xl font-bold text-blue-400" id="ui-project-name">Noise Engine Controller</h1>
                <p class="text-gray-400 text-sm mt-1">Live Audio Streaming & Model Training Dashboard</p>
            </div>
            <div class="flex items-center space-x-3">
                <span class="relative flex h-3 w-3">
                  <span id="status-ping" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span id="status-dot" class="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                </span>
                <span id="status-text" class="text-sm text-gray-300 font-mono">Status: Idle</span>
            </div>
        </header>

        <!-- Tabs Navigation -->
        <div class="flex space-x-6 border-b border-gray-700 mb-6">
            <button onclick="switchTab('dashboard')" id="tab-dashboard" class="tab-btn active pb-2 text-gray-400 hover:text-white font-medium transition">Dashboard</button>
            <button onclick="switchTab('downloader')" id="tab-downloader" class="tab-btn pb-2 text-gray-400 hover:text-white font-medium transition">Stream / Downloader</button>
            <button onclick="switchTab('config')" id="tab-config" class="tab-btn pb-2 text-gray-400 hover:text-white font-medium transition">Model Parameters</button>
            <button onclick="switchTab('inference')" id="tab-inference" class="tab-btn pb-2 text-gray-400 hover:text-white font-medium transition">Inference / Test</button>
        </div>

        <!-- TAB: Dashboard -->
        <div id="content-dashboard" class="tab-content active">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
                <div class="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
                    <h3 class="text-gray-400 text-xs uppercase tracking-wider mb-1">Training Queue</h3>
                    <p class="text-3xl font-bold text-white" id="stat-queue">0</p>
                    <p class="text-xs text-gray-500 mt-1">Audio files buffered</p>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
                    <h3 class="text-gray-400 text-xs uppercase tracking-wider mb-1">Epochs / Steps</h3>
                    <p class="text-3xl font-bold text-white" id="stat-epochs">0</p>
                    <p class="text-xs text-gray-500 mt-1">Completed iterations</p>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700 flex flex-col justify-center">
                    <button id="btn-toggle-train" onclick="toggleTraining()" class="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded shadow transition">
                        Start Training Engine
                    </button>
                </div>
                <div class="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700 flex flex-col justify-center">
                    <button onclick="downloadWeights()" class="w-full py-3 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded shadow transition flex items-center justify-center space-x-2">
                        <span>Download Weights</span>
                        <span class="text-xs bg-purple-800 px-2 py-1 rounded">.ZIP</span>
                    </button>
                </div>
            </div>

            <div class="bg-gray-800 rounded-lg shadow-lg border border-gray-700 overflow-hidden">
                <div class="bg-gray-900 px-4 py-2 border-b border-gray-700 flex justify-between">
                    <span class="text-xs text-gray-400 font-mono">System Console logs</span>
                </div>
                <div class="p-4 h-64 overflow-y-auto font-mono text-sm space-y-1" id="log-container">
                    <!-- Logs injected via JS -->
                </div>
            </div>
        </div>

        <!-- TAB: Downloader -->
        <div id="content-downloader" class="tab-content">
            <div class="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
                <h2 class="text-xl font-bold mb-4">Add Audio Stream</h2>
                <p class="text-sm text-gray-400 mb-4">Paste URLs from YouTube, Internet Radio, or Audio Datasets. The background thread will pull the streams while the model trains simultaneously.</p>
                
                <div class="flex space-x-4">
                    <input type="text" id="stream-url" placeholder="https://example.com/audio-stream" class="flex-1 bg-gray-900 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:border-blue-500">
                    <button onclick="addStream()" class="bg-green-600 hover:bg-green-500 px-6 py-2 rounded font-bold transition">Queue Audio</button>
                </div>
            </div>
        </div>

        <!-- TAB: Configuration -->
        <div id="content-config" class="tab-content">
            <div class="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
                <h2 class="text-xl font-bold mb-6">Model Architecture & Settings</h2>
                
                <form id="config-form" onsubmit="saveConfig(event)" class="space-y-6">
                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-1">Project Name</label>
                        <input type="text" id="cfg-name" class="w-full bg-gray-900 border border-gray-600 rounded px-4 py-2 focus:outline-none focus:border-blue-500">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-1">Model Architecture</label>
                        <select id="cfg-arch" class="w-full bg-gray-900 border border-gray-600 rounded px-4 py-2 focus:outline-none focus:border-blue-500">
                            <option value="1B Dense">1B Parameter (Dense) - Good for testing</option>
                            <option value="4B-A1B MoE">4B Total / 1B Active (Mixture of Experts)</option>
                        </select>
                    </div>

                    <div class="flex items-center">
                        <input type="checkbox" id="cfg-moe" class="h-5 w-5 rounded bg-gray-900 border-gray-600 text-blue-500 focus:ring-blue-500 focus:ring-opacity-25">
                        <label class="ml-3 block text-sm font-medium text-gray-300">
                            Enable MoE Parameter Activation (Sparse Routing)
                        </label>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-1">Max Audio Output Longevity (Seconds)</label>
                        <input type="number" id="cfg-longevity" class="w-full bg-gray-900 border border-gray-600 rounded px-4 py-2 focus:outline-none focus:border-blue-500">
                        <p class="text-xs text-gray-500 mt-1">Crops continuous streams to this length to prevent VRAM overflow.</p>
                    </div>
                    
                    <div class="pt-6 border-t border-gray-700">
                        <h3 class="text-lg font-bold mb-4 text-purple-400">Hugging Face Integration</h3>
                        <div class="space-y-4">
                            <div>
                                <label class="block text-sm font-medium text-gray-400 mb-1">HF Repository ID</label>
                                <input type="text" id="cfg-hf-repo" placeholder="username/noise-engine-1B" class="w-full bg-gray-900 border border-gray-600 rounded px-4 py-2 focus:outline-none focus:border-purple-500">
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-400 mb-1">HF Access Token (Write)</label>
                                <input type="password" id="cfg-hf-token" placeholder="hf_..." class="w-full bg-gray-900 border border-gray-600 rounded px-4 py-2 focus:outline-none focus:border-purple-500">
                            </div>
                        </div>
                    </div>

                    <div class="pt-4 border-t border-gray-700 flex justify-end">
                        <button type="submit" class="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded font-bold transition">Apply Changes</button>
                    </div>
                </form>
            </div>
        </div>
        
        <!-- TAB: Inference -->
        <div id="content-inference" class="tab-content">
            <div class="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
                <h2 class="text-xl font-bold mb-4">Run Trained Model</h2>
                <p class="text-sm text-gray-400 mb-6">Generate audio output using your current model weights. (Currently generates a test tone to verify pipeline).</p>
                
                <div class="space-y-6">
                    <button onclick="runInference()" class="w-full bg-green-600 hover:bg-green-500 py-3 rounded font-bold transition">
                        Generate Audio Output
                    </button>
                    
                    <div id="audio-player-container" class="hidden mt-6 p-4 bg-gray-900 rounded border border-gray-700">
                        <h3 class="text-sm text-gray-400 mb-2">Generated Output:</h3>
                        <audio id="audio-player" controls class="w-full">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <script>
        // Tab switching logic
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            
            document.getElementById(`content-${tabId}`).classList.add('active');
            document.getElementById(`tab-${tabId}`).classList.add('active');
        }

        // Add Stream logic
        async function addStream() {
            const urlInput = document.getElementById('stream-url');
            const url = urlInput.value.trim();
            if (!url) return;
            
            await fetch('/api/add_stream', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ url: url })
            });
            urlInput.value = '';
            // Auto switch back to dashboard to see logs
            switchTab('dashboard');
        }

        // Toggle Training Engine
        async function toggleTraining() {
            await fetch('/api/toggle_training', { method: 'POST' });
        }

        // Download Weights ZIP
        function downloadWeights() {
            window.location.href = '/api/download_weights';
        }

        // Run Inference
        async function runInference() {
            const btn = document.getElementById('btn-run-infer');
            btn.innerText = "Processing Audio...";
            btn.classList.add('opacity-50', 'cursor-not-allowed');
            
            try {
                const res = await fetch('/api/run_inference', { method: 'POST' });
                const data = await res.json();
                
                if (data.status === "success") {
                    const player = document.getElementById('output-audio-player');
                    // Add timestamp to prevent browser caching the audio
                    player.src = data.audio_url + '?t=' + new Date().getTime();
                    document.getElementById('audio-result-container').classList.remove('hidden');
                }
            } catch (err) {
                console.error("Inference failed:", err);
            }
            
            btn.innerText = "Generate Output Audio";
            btn.classList.remove('opacity-50', 'cursor-not-allowed');
        }

        // Save Configuration
        async function saveConfig(e) {
            e.preventDefault();
            const config = {
                project_name: document.getElementById('cfg-name').value,
                model_architecture: document.getElementById('cfg-arch').value,
                moe_activation_params: document.getElementById('cfg-moe').checked,
                max_song_longevity_sec: parseInt(document.getElementById('cfg-longevity').value),
                hf_repo_id: document.getElementById('cfg-hf-repo').value,
                hf_token: document.getElementById('cfg-hf-token').value
            };
            
            await fetch('/api/update_params', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(config)
            });
            alert("Settings applied securely to the Engine.");
        }
        
        // Run Inference
        async function runInference() {
            const playerContainer = document.getElementById('audio-player-container');
            const player = document.getElementById('audio-player');
            
            // Show player and fetch the dynamically generated audio
            playerContainer.classList.remove('hidden');
            player.src = '/api/infer?t=' + new Date().getTime(); // Prevent caching
            player.play();
        }

        // Polling state from backend
        async function fetchState() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                
                // Update stats
                document.getElementById('stat-queue').innerText = data.files_ready;
                document.getElementById('stat-epochs').innerText = data.epochs_completed;
                document.getElementById('ui-project-name').innerText = data.project_name;
                
                // Update training button & indicator
                const btn = document.getElementById('btn-toggle-train');
                const dot = document.getElementById('status-dot');
                const ping = document.getElementById('status-ping');
                const txt = document.getElementById('status-text');

                if (data.is_training) {
                    btn.innerText = "Stop Training Engine";
                    btn.classList.replace('bg-blue-600', 'bg-red-600');
                    btn.classList.replace('hover:bg-blue-500', 'hover:bg-red-500');
                    dot.classList.replace('bg-red-500', 'bg-green-500');
                    ping.classList.replace('bg-red-400', 'bg-green-400');
                    txt.innerText = "Status: Training Active";
                } else {
                    btn.innerText = "Start Training Engine";
                    btn.classList.replace('bg-red-600', 'bg-blue-600');
                    btn.classList.replace('hover:bg-red-500', 'hover:bg-blue-500');
                    dot.classList.replace('bg-green-500', 'bg-red-500');
                    ping.classList.replace('bg-green-400', 'bg-red-400');
                    txt.innerText = "Status: Idle";
                }

                // Update Config form values (only if not currently focused)
                if (document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "SELECT") {
                    document.getElementById('cfg-name').value = data.project_name;
                    document.getElementById('cfg-arch').value = data.model_architecture;
                    document.getElementById('cfg-moe').checked = data.moe_activation_params;
                    document.getElementById('cfg-longevity').value = data.max_song_longevity_sec;
                    document.getElementById('cfg-hf-repo').value = data.hf_repo_id;
                    // Don't auto-update password field to prevent clearing while typing
                    if(data.hf_token && document.getElementById('cfg-hf-token').value === "") {
                        document.getElementById('cfg-hf-token').value = data.hf_token;
                    }
                }

                // Update Logs
                const logContainer = document.getElementById('log-container');
                logContainer.innerHTML = '';
                data.logs.forEach(log => {
                    const el = document.createElement('div');
                    // Format output for specific keywords
                    if (log.includes("Error")) el.className = "text-red-400";
                    else if (log.includes("Successfully") || log.includes("Finished")) el.className = "text-green-400";
                    else el.className = "text-gray-300";
                    el.innerText = log;
                    logContainer.appendChild(el);
                });

            } catch (err) {
                console.error("Connection lost:", err);
            }
        }

        // Poll every 1 second
        setInterval(fetchState, 1000);
        fetchState(); // Initial fetch
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Flask Routes / API
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Serves the main single-page application UI."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns the current state of the backend to the UI frontend."""
    return jsonify(state)

@app.route('/api/add_stream', methods=['POST'])
def add_stream():
    """Adds a new audio URL to the download/stream queue."""
    data = request.json
    url = data.get('url', '')
    if url:
        stream_queue.put(url)
        log_msg(f"Added URL to streaming queue: {url}")
        return jsonify({"status": "queued"})
    return jsonify({"status": "error", "message": "No URL provided"}), 400

@app.route('/api/toggle_training', methods=['POST'])
def toggle_training():
    """Starts or stops the consumer training thread processing."""
    state["is_training"] = not state["is_training"]
    status_str = "Resumed" if state["is_training"] else "Paused"
    log_msg(f"User commanded Engine: {status_str}")
    return jsonify({"is_training": state["is_training"]})

@app.route('/api/update_params', methods=['POST'])
def update_params():
    """Updates the model architecture and configuration parameters."""
    data = request.json
    state["project_name"] = data.get("project_name", state["project_name"])
    state["model_architecture"] = data.get("model_architecture", state["model_architecture"])
    state["moe_activation_params"] = data.get("moe_activation_params", state["moe_activation_params"])
    state["max_song_longevity_sec"] = data.get("max_song_longevity_sec", state["max_song_longevity_sec"])
    state["hf_repo_id"] = data.get("hf_repo_id", state["hf_repo_id"])
    state["hf_token"] = data.get("hf_token", state["hf_token"])
    
    log_msg("Engine configuration parameters updated via Dashboard.")
    if state["hf_repo_id"]:
        log_msg(f"HF Integration primed for repo: {state['hf_repo_id']}")
        
    return jsonify({"status": "updated"})

@app.route('/api/download_weights', methods=['GET'])
def download_weights():
    """Generates a ZIP file containing the model weights and config dynamically."""
    log_msg("Generating model weights ZIP package for download...")
    
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Create a Hugging Face compatible config.json
        config_data = f"""{{
    "architectures": ["NoiseEngineModel"],
    "model_type": "{state['model_architecture']}",
    "moe_active": {str(state['moe_activation_params']).lower()},
    "max_audio_length": {state['max_song_longevity_sec']}
}}"""
        zf.writestr('config.json', config_data)
        
        # Simulate standard PyTorch weight file
        dummy_weights = b"This is a placeholder for the actual safetensors or pytorch_model.bin weight data."
        zf.writestr('pytorch_model.bin', dummy_weights)
        
        # Add a README.md
        readme = f"# {state['project_name']}\nA custom noise engine exported from the training dashboard."
        zf.writestr('README.md', readme)

    memory_file.seek(0)
    log_msg("ZIP package ready. Sending to client.")
    
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='noise_engine_weights.zip'
    )

@app.route('/api/infer', methods=['GET'])
def infer_audio():
    """Generates a valid .wav file (sine wave) to simulate model inference output."""
    log_msg("Running inference step. Generating audio output...")
    
    # Generate a 2-second 440Hz sine wave as a placeholder for model output
    sample_rate = 44100
    duration = 2.0
    frequency = 440.0
    
    memory_file = io.BytesIO()
    with wave.open(memory_file, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for i in range(int(sample_rate * duration)):
            value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)
            
    memory_file.seek(0)
    log_msg("Inference complete. Streaming audio to client.")
    
    return send_file(
        memory_file,
        mimetype='audio/wav'
    )

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("="*60)
    print("Starting Noise Engine Server...")
    print("Access the dashboard at: http://localhost:5000")
    print("="*60)
    # Host on 0.0.0.0 to be accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
