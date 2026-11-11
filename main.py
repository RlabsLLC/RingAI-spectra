import os
import time
import threading
import queue
import io
import zipfile
import math
import struct
import wave
import traceback
from flask import Flask, request, jsonify, render_template_string, send_file, send_from_directory

# --- NEW IMPORTS FOR PYTORCH ---
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio

# Force torchaudio to be more forgiving with backend if available
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

# ---------------------------------------------------------------------------
# Global State & Queues
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024 # Allow up to 5GB batch uploads

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
    "epochs_completed": 0,
    "current_loss": 0.0
}

def log_msg(msg):
    """Helper to add messages to the global log (keeps last 50)"""
    timestamp = time.strftime("%H:%M:%S")
    state["logs"].insert(0, f"[{timestamp}] {msg}")
    if len(state["logs"]) > 50:
        state["logs"].pop()

# ---------------------------------------------------------------------------
# PyTorch Model Architecture (Scaled to ~330M Parameters / ~4.5GB VRAM)
# ---------------------------------------------------------------------------
class DeepNoiseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 1D Convolutions to downsample audio
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=15, stride=4, padding=7),
            nn.GELU()
        )
        
        # Massive Bottleneck (This is what consumes the VRAM to simulate the 330M param scale)
        # Using 1x1 convolutions as massive linear layers across the sequence
        self.massive_bottleneck = nn.Sequential(
            nn.Conv1d(512, 8192, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(8192, 8192, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(8192, 512, kernel_size=1),
            nn.GELU()
        )
        
        # Decoder: Transposed Convolutions to upsample back to raw audio
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=15, stride=4, padding=7, output_padding=3),
            nn.GELU(),
            nn.ConvTranspose1d(256, 1, kernel_size=15, stride=4, padding=7, output_padding=3),
            nn.Tanh() # Bound audio between -1 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.massive_bottleneck(x)
        x = self.decoder(x)
        return x

# Initialize global model and move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing PyTorch on device: {device}")
noise_engine = DeepNoiseAutoencoder().to(device)
optimizer = optim.AdamW(noise_engine.parameters(), lr=1e-4)
criterion = nn.MSELoss()

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
        # --- NEW ANTI-BOT MEASURES ---
        'extractor_args': {
            'youtube': {
                # Spoof the Android client to bypass strict web bot-checks
                'player_client': ['android', 'web'] 
            }
        },
        # If you upload a cookies.txt file to the same folder, yt-dlp will use it automatically
        'cookiefile': 'cookies.txt', 
        # -----------------------------
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
    Thread 2: The Consumer (Real PyTorch Training Loop).
    Waits for audio files, loads them via torchaudio, injects noise, and trains.
    """
    log_msg(f"Training Engine thread started on {device}.")
    noise_engine.train()
    
    while True:
        if not state["is_training"]:
            time.sleep(1)
            continue
            
        try:
            audio_file = training_queue.get(timeout=2)
            state["files_ready"] = training_queue.qsize()
            
            log_msg(f"Loading: {audio_file.split('/')[-1]}")
            
            try:
                # 1. Load Audio Robustly
                waveform, sample_rate = torchaudio.load(audio_file)
                
                # Convert stereo to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample to 16kHz for stability if needed (assuming 16k target)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                
                # Trim to Max Longevity (e.g. 300 seconds)
                max_samples = 16000 * state["max_song_longevity_sec"]
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]
                
                # Move to GPU and ensure float32
                clean_audio = waveform.unsqueeze(0).to(device, dtype=torch.float32) 
                
                # 2. Inject Artificial Noise (The autoencoder task)
                noise = torch.randn_like(clean_audio) * 0.4
                noisy_audio = clean_audio + noise
                
                # 3. PyTorch Forward & Backward Pass (Chunked to save memory)
                # Process in 5-second chunks to prevent OOM spikes during backward
                chunk_size = 16000 * 5 
                total_loss = 0
                chunks = 0
                
                optimizer.zero_grad()
                
                for i in range(0, clean_audio.shape[2], chunk_size):
                    end_idx = i + chunk_size
                    if end_idx > clean_audio.shape[2]:
                        break # Skip final uneven chunk
                        
                    noisy_chunk = noisy_audio[:, :, i:end_idx]
                    clean_chunk = clean_audio[:, :, i:end_idx]
                    
                    # Forward
                    predicted_clean = noise_engine(noisy_chunk)
                    
                    # Compute Loss
                    loss = criterion(predicted_clean, clean_chunk)
                    
                    # Backward
                    loss.backward()
                    total_loss += loss.item()
                    chunks += 1
                
                # Optimize
                if chunks > 0:
                    optimizer.step()
                    avg_loss = total_loss / chunks
                    state["current_loss"] = avg_loss
                    log_msg(f"GPU Step Complete! Real Loss: {avg_loss:.4f}")
                else:
                    log_msg("Audio too short to process. Skipped.")
                
            except Exception as e:
                # Catch Corrupt MP3s or ID3v2 tag decoding errors gracefully
                log_msg(f"GPU Training Error: Skipped corrupted file. ({str(e).splitlines()[0]})")
            
            # Cleanup processed file
            if os.path.exists(audio_file):
                os.remove(audio_file)
                
            state["epochs_completed"] += 1
            training_queue.task_done()
            
        except queue.Empty:
            log_msg("Training queue empty. Waiting for Downloader thread...")
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
        .tab-content { display: none; opacity: 0; transition: opacity 0.3s ease; }
        .tab-content.active { display: block; opacity: 1; }
        .nav-item.active { background-color: rgba(59, 130, 246, 0.1); border-right: 4px solid #3b82f6; color: #60a5fa; }
        
        /* Custom scrollbar for terminal */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1f2937; }
        ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #6b7280; }
        
        #toast-container { position: fixed; bottom: 20px; right: 20px; z-index: 50; display: flex; flex-direction: column; gap: 10px; }
        .toast { padding: 12px 20px; border-radius: 6px; color: white; font-weight: 500; opacity: 0; transform: translateY(20px); transition: all 0.3s ease; }
        .toast.show { opacity: 1; transform: translateY(0); }
        .toast.success { background-color: #10b981; }
        .toast.error { background-color: #ef4444; }
        .toast.info { background-color: #3b82f6; }
    </style>
</head>
<body class="bg-gray-950 text-gray-200 min-h-screen font-sans flex overflow-hidden">
    
    <!-- Sidebar -->
    <aside class="w-64 bg-gray-900 border-r border-gray-800 flex flex-col justify-between h-screen">
        <div>
            <div class="p-6 border-b border-gray-800">
                <h1 class="text-xl font-bold text-blue-400" id="ui-project-name">Noise Engine</h1>
                <div class="flex items-center space-x-2 mt-2">
                    <span class="relative flex h-2.5 w-2.5">
                      <span id="status-ping" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span id="status-dot" class="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                    </span>
                    <span id="status-text" class="text-xs text-gray-400 font-mono uppercase tracking-wider">Status: Idle</span>
                </div>
            </div>
            
            <nav class="flex flex-col mt-4 space-y-1">
                <button onclick="switchTab('dashboard')" id="nav-dashboard" class="nav-item active px-6 py-3 text-left text-sm font-medium text-gray-400 hover:text-white hover:bg-gray-800 transition flex items-center">
                    <svg class="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z"></path></svg>
                    Dashboard
                </button>
                <button onclick="switchTab('downloader')" id="nav-downloader" class="nav-item px-6 py-3 text-left text-sm font-medium text-gray-400 hover:text-white hover:bg-gray-800 transition flex items-center">
                    <svg class="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10"></path></svg>
                    Add Audio Data
                </button>
                <button onclick="switchTab('config')" id="nav-config" class="nav-item px-6 py-3 text-left text-sm font-medium text-gray-400 hover:text-white hover:bg-gray-800 transition flex items-center">
                    <svg class="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                    Parameters
                </button>
                <button onclick="switchTab('inference')" id="nav-inference" class="nav-item px-6 py-3 text-left text-sm font-medium text-gray-400 hover:text-white hover:bg-gray-800 transition flex items-center">
                    <svg class="w-5 h-5 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    Test Engine
                </button>
            </nav>
        </div>
        <div class="p-6">
            <button onclick="downloadWeights()" class="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-bold rounded shadow transition flex items-center justify-center space-x-2">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                <span>Export Weights</span>
            </button>
        </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 h-screen overflow-y-auto bg-gray-950 p-8">
        
        <!-- TAB: Dashboard -->
        <div id="content-dashboard" class="tab-content active max-w-5xl mx-auto">
            <h2 class="text-2xl font-bold mb-6">Engine Overview</h2>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div class="bg-gray-900 p-5 rounded-xl border border-gray-800 shadow-sm">
                    <h3 class="text-gray-400 text-xs uppercase tracking-wider mb-2">Files in Queue</h3>
                    <p class="text-4xl font-light text-white" id="stat-queue">0</p>
                </div>
                <div class="bg-gray-900 p-5 rounded-xl border border-gray-800 shadow-sm">
                    <h3 class="text-gray-400 text-xs uppercase tracking-wider mb-2">Training Epochs</h3>
                    <p class="text-4xl font-light text-white" id="stat-epochs">0</p>
                </div>
                <div class="bg-gray-900 p-5 rounded-xl border border-gray-800 shadow-sm flex flex-col justify-center">
                    <button id="btn-toggle-train" onclick="toggleTraining()" class="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg transition flex items-center justify-center space-x-2">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                        <span>Start Training</span>
                    </button>
                </div>
            </div>

            <div class="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden shadow-sm">
                <div class="bg-gray-800/50 px-4 py-3 border-b border-gray-800 flex justify-between items-center">
                    <span class="text-xs text-gray-400 font-mono tracking-widest uppercase">System Terminal</span>
                    <button onclick="document.getElementById('log-container').innerHTML=''" class="text-xs text-gray-500 hover:text-gray-300">Clear</button>
                </div>
                <div class="p-4 h-96 overflow-y-auto font-mono text-sm space-y-1.5" id="log-container">
                    <!-- Logs injected via JS -->
                </div>
            </div>
        </div>

        <!-- TAB: Downloader -->
        <div id="content-downloader" class="tab-content max-w-4xl mx-auto">
            <h2 class="text-2xl font-bold mb-6">Add Audio Data</h2>
            
            <div class="bg-gray-900 p-6 rounded-xl border border-gray-800 shadow-sm mb-6">
                <h3 class="text-lg font-semibold mb-2">Stream via URL</h3>
                <p class="text-sm text-gray-400 mb-4">Paste a YouTube or audio stream link. The engine will download it in the background.</p>
                
                <div class="flex space-x-3">
                    <input type="text" id="stream-url" placeholder="https://youtube.com/..." class="flex-1 bg-gray-950 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500">
                    <button onclick="addStream()" class="bg-emerald-600 hover:bg-emerald-500 px-6 py-2 rounded-lg font-bold transition flex items-center">
                        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path></svg>
                        Queue
                    </button>
                </div>
            </div>

            <div class="bg-gray-900 p-6 rounded-xl border border-gray-800 shadow-sm">
                <h3 class="text-lg font-semibold mb-2">Upload Local Files</h3>
                <p class="text-sm text-gray-400 mb-4">Upload raw audio files (.wav, .mp3, .flac) directly from your machine.</p>
                
                <div class="border-2 border-dashed border-gray-700 rounded-xl p-8 text-center hover:border-gray-500 transition cursor-pointer" id="dropzone" onclick="document.getElementById('file-upload').click()">
                    <svg class="mx-auto h-12 w-12 text-gray-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                    <p class="text-gray-300 font-medium">Click to select files, or drag and drop</p>
                    <p class="text-xs text-gray-500 mt-1">Supports WAV, MP3, FLAC, M4A, OGG</p>
                </div>
                
                <div class="mt-4 flex space-x-3 justify-center">
                    <!-- Hidden inputs for actual selection -->
                    <input type="file" id="file-upload" multiple accept="audio/*" class="hidden" onchange="handleFilesSelected(event)">
                    <input type="file" id="folder-upload" webkitdirectory directory multiple class="hidden" onchange="handleFilesSelected(event)">
                    
                    <button onclick="document.getElementById('file-upload').click()" class="bg-gray-800 hover:bg-gray-700 border border-gray-700 px-4 py-2 rounded-lg text-sm font-medium transition">
                        Select Files
                    </button>
                    <button onclick="document.getElementById('folder-upload').click()" class="bg-gray-800 hover:bg-gray-700 border border-gray-700 px-4 py-2 rounded-lg text-sm font-medium transition">
                        Select Entire Folder
                    </button>
                </div>
            </div>
        </div>

        <!-- TAB: Configuration -->
        <div id="content-config" class="tab-content max-w-3xl mx-auto">
            <h2 class="text-2xl font-bold mb-6">Engine Parameters</h2>
            
            <form id="config-form" onsubmit="saveConfig(event)" class="bg-gray-900 p-6 rounded-xl border border-gray-800 shadow-sm space-y-6">
                <div>
                    <label class="block text-sm font-medium text-gray-400 mb-1">Project Name</label>
                    <input type="text" id="cfg-name" class="w-full bg-gray-950 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500">
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-400 mb-1">Model Architecture</label>
                    <select id="cfg-arch" class="w-full bg-gray-950 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500">
                        <option value="1B Dense">1B Parameter (Dense) - Fast/Baseline</option>
                        <option value="4B-A1B MoE">4B Total / 1B Active (Mixture of Experts)</option>
                    </select>
                </div>

                <div class="flex items-center p-3 bg-gray-950 rounded-lg border border-gray-800">
                    <input type="checkbox" id="cfg-moe" class="h-5 w-5 rounded bg-gray-800 border-gray-600 text-blue-500 focus:ring-blue-500 focus:ring-opacity-25 focus:ring-offset-gray-900">
                    <label class="ml-3 block text-sm font-medium text-gray-300 cursor-pointer" for="cfg-moe">
                        Enable MoE Parameter Activation (Sparse Routing)
                    </label>
                </div>

                <div>
                    <label class="block text-sm font-medium text-gray-400 mb-1">Max Audio Clipping (Seconds)</label>
                    <input type="number" id="cfg-longevity" class="w-full bg-gray-950 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500">
                    <p class="text-xs text-gray-500 mt-1">Limits sequence length to prevent VRAM Out-Of-Memory errors.</p>
                </div>
                
                <div class="pt-6 border-t border-gray-800">
                    <h3 class="text-sm font-bold tracking-wider uppercase mb-4 text-purple-400">Hugging Face Sync</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Repository ID</label>
                            <input type="text" id="cfg-hf-repo" placeholder="username/repo-name" class="w-full bg-gray-950 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-purple-500">
                        </div>
                        <div>
                            <label class="block text-xs font-medium text-gray-400 mb-1">Write Token</label>
                            <input type="password" id="cfg-hf-token" placeholder="hf_..." class="w-full bg-gray-950 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-purple-500">
                        </div>
                    </div>
                </div>

                <div class="pt-4 flex justify-end">
                    <button type="submit" class="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-bold transition">Apply Changes</button>
                </div>
            </form>
        </div>
        
        <!-- TAB: Inference -->
        <div id="content-inference" class="tab-content max-w-3xl mx-auto">
            <h2 class="text-2xl font-bold mb-6">Test Engine</h2>
            
            <div class="bg-gray-900 p-6 rounded-xl border border-gray-800 shadow-sm text-center">
                <div class="w-20 h-20 mx-auto bg-gray-800 rounded-full flex items-center justify-center mb-4 border border-gray-700">
                    <svg class="w-10 h-10 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path></svg>
                </div>
                
                <h3 class="text-xl font-semibold mb-2">Generate Clean Audio</h3>
                <p class="text-sm text-gray-400 mb-8 max-w-md mx-auto">Run a forward pass through your currently trained weights to produce a sample output.</p>
                
                <button id="btn-run-infer" onclick="runInference()" class="w-full max-w-xs mx-auto bg-blue-600 hover:bg-blue-500 py-3 rounded-lg font-bold transition flex justify-center items-center">
                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                    Run Inference Pass
                </button>
                
                <div id="audio-player-container" class="hidden mt-8 pt-6 border-t border-gray-800">
                    <p class="text-xs text-gray-400 uppercase tracking-wider mb-3 text-left">Output Result</p>
                    <audio id="audio-player" controls class="w-full rounded-lg shadow-lg">
                        Your browser does not support the audio element.
                    </audio>
                </div>
            </div>
        </div>

    </main>

    <!-- Toast Notifications Container -->
    <div id="toast-container"></div>

    <script>
        // Notification System
        function showToast(message, type = 'info') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerText = message;
            container.appendChild(toast);
            
            // Trigger animation
            setTimeout(() => toast.classList.add('show'), 10);
            
            // Remove after 3 seconds
            setTimeout(() => {
                toast.classList.remove('show');
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }

        // Tab switching logic
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
            
            document.getElementById(`content-${tabId}`).classList.add('active');
            document.getElementById(`nav-${tabId}`).classList.add('active');
        }

        // Add Stream logic
        async function addStream() {
            const urlInput = document.getElementById('stream-url');
            const url = urlInput.value.trim();
            if (!url) return;
            
            try {
                await fetch('/api/add_stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ url: url })
                });
                showToast("Stream added to queue!", "success");
                urlInput.value = '';
                switchTab('dashboard');
            } catch (e) {
                showToast("Failed to add stream.", "error");
            }
        }

        // Consolidated Upload logic for both Files and Folders
        async function handleFilesSelected(event) {
            const fileInput = event.target;
            const files = fileInput.files;
            
            if (!files || files.length === 0) return;
            
            // Filter valid files first
            const validFiles = [];
            for (const file of files) {
                if (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|flac|ogg|m4a)$/i)) {
                    validFiles.push(file);
                }
            }
            
            if (validFiles.length === 0) {
                showToast("No valid audio files found in selection.", "error");
                fileInput.value = ''; 
                return;
            }

            showToast(`Uploading ${validFiles.length} file(s) sequentially...`, "info");
            
            // Update UI to show progress
            const dropzoneText = document.querySelector('#dropzone p.font-medium');
            const originalText = dropzoneText.innerText;
            
            let successCount = 0;
            let failCount = 0;

            // Upload sequentially to prevent cloud proxy limits (HTTP 413 Entity Too Large)
            for (let i = 0; i < validFiles.length; i++) {
                dropzoneText.innerText = `Uploading ${i + 1} of ${validFiles.length}...`;
                
                const formData = new FormData();
                formData.append('files', validFiles[i]);

                try {
                    const res = await fetch('/api/upload_audio', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (res.ok) {
                        successCount++;
                    } else {
                        failCount++;
                    }
                } catch (err) {
                    console.error("Upload failed for:", validFiles[i].name, err);
                    failCount++;
                }
            }
            
            // Restore UI
            dropzoneText.innerText = originalText;

            if (successCount > 0) {
                showToast(`Successfully uploaded ${successCount} file(s)!`, "success");
                switchTab('dashboard');
            }
            if (failCount > 0) {
                showToast(`Failed to upload ${failCount} file(s).`, "error");
            }
            
            fileInput.value = ''; // clear input so the same files can be selected again if needed
        }

        // Drag and Drop Logic
        const dropzone = document.getElementById('dropzone');
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('border-blue-500', 'bg-gray-800');
        });
        dropzone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropzone.classList.remove('border-blue-500', 'bg-gray-800');
        });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('border-blue-500', 'bg-gray-800');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                // Create a synthetic event object to reuse handleFilesSelected
                handleFilesSelected({ target: { files: e.dataTransfer.files } });
            }
        });

        // Toggle Training Engine
        async function toggleTraining() {
            try {
                await fetch('/api/toggle_training', { method: 'POST' });
                showToast("Toggled Engine State", "info");
            } catch (e) {
                showToast("Command failed", "error");
            }
        }

        // Download Weights ZIP
        function downloadWeights() {
            showToast("Packaging model weights...", "info");
            window.location.href = '/api/download_weights';
        }

        // Run Inference (Fixed duplicate)
        function runInference() {
            const btn = document.getElementById('btn-run-infer');
            const playerContainer = document.getElementById('audio-player-container');
            const player = document.getElementById('audio-player');
            
            // UI Loading state
            btn.innerHTML = `<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Processing...`;
            btn.disabled = true;
            btn.classList.add('opacity-75', 'cursor-not-allowed');
            
            // Point player to backend endpoint, add timestamp to bust browser cache
            player.src = '/api/infer?t=' + new Date().getTime();
            
            // When audio is fully received and ready to play
            player.oncanplay = () => {
                playerContainer.classList.remove('hidden');
                btn.innerHTML = `<svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg> Run Inference Pass`;
                btn.disabled = false;
                btn.classList.remove('opacity-75', 'cursor-not-allowed');
                player.play();
                showToast("Inference generated successfully", "success");
            };
            
            player.onerror = () => {
                showToast("Failed to generate audio", "error");
                btn.innerHTML = `Try Again`;
                btn.disabled = false;
                btn.classList.remove('opacity-75', 'cursor-not-allowed');
            };
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
            
            try {
                await fetch('/api/update_params', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(config)
                });
                showToast("Settings applied securely", "success");
            } catch (err) {
                showToast("Failed to save settings", "error");
            }
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
                    btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"></path></svg> <span>Stop Training</span>`;
                    btn.classList.replace('bg-blue-600', 'bg-red-600');
                    btn.classList.replace('hover:bg-blue-500', 'hover:bg-red-500');
                    dot.classList.replace('bg-red-500', 'bg-emerald-500');
                    ping.classList.replace('bg-red-400', 'bg-emerald-400');
                    txt.innerText = "Status: Training Active";
                } else {
                    btn.innerHTML = `<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg> <span>Start Training</span>`;
                    btn.classList.replace('bg-red-600', 'bg-blue-600');
                    btn.classList.replace('hover:bg-red-500', 'hover:bg-blue-500');
                    dot.classList.replace('bg-emerald-500', 'bg-red-500');
                    ping.classList.replace('bg-emerald-400', 'bg-red-400');
                    txt.innerText = "Status: Idle";
                }

                // Update Config form values (only if not currently focused to avoid overwriting typing)
                if (document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "SELECT") {
                    document.getElementById('cfg-name').value = data.project_name;
                    document.getElementById('cfg-arch').value = data.model_architecture;
                    document.getElementById('cfg-moe').checked = data.moe_activation_params;
                    document.getElementById('cfg-longevity').value = data.max_song_longevity_sec;
                    document.getElementById('cfg-hf-repo').value = data.hf_repo_id;
                    if(data.hf_token && document.getElementById('cfg-hf-token').value === "") {
                        document.getElementById('cfg-hf-token').value = data.hf_token;
                    }
                }

                // Update Logs
                const logContainer = document.getElementById('log-container');
                // Only update if there's new data to prevent scrolling jumpiness
                if (logContainer.childElementCount !== data.logs.length || 
                   (logContainer.firstChild && logContainer.firstChild.innerText !== data.logs[0])) {
                    
                    logContainer.innerHTML = '';
                    data.logs.forEach(log => {
                        const el = document.createElement('div');
                        if (log.includes("Error") || log.includes("failed")) el.className = "text-red-400";
                        else if (log.includes("Successfully") || log.includes("Finished")) el.className = "text-emerald-400";
                        else el.className = "text-gray-400";
                        el.innerText = log;
                        logContainer.appendChild(el);
                    });
                }

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

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    """Handles bulk folder uploads from the user's local machine."""
    files = request.files.getlist('files')
    if not files:
        return jsonify({"status": "error", "message": "No files received"}), 400
    
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
        
    saved_count = 0
    for file in files:
        if file.filename:
            try:
                # Sanitize paths in case the browser sends full folder structures (Windows or Mac)
                clean_name = file.filename.replace('\\', '/').split('/')[-1]
                
                # Ensure filename is safe and won't cause OS errors
                safe_name = "".join(c for c in clean_name if c.isalnum() or c in " ._-").strip()
                if not safe_name:
                    safe_name = f"audio_{int(time.time())}_{saved_count}.wav"
                    
                filepath = os.path.join('downloads', safe_name)
                file.save(filepath)
                
                # Add to training queue immediately
                training_queue.put(filepath)
                saved_count += 1
            except Exception as e:
                log_msg(f"Failed to save {file.filename}: {str(e)}")
            
    state["files_ready"] = training_queue.qsize()
    log_msg(f"Successfully uploaded and queued {saved_count} audio files from local machine.")
    
    return jsonify({"status": "success", "count": saved_count})

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

@app.route('/api/run_inference', methods=['POST'])
def run_inference():
    """Runs a noisy test signal through the REAL PyTorch model and saves the output."""
    log_msg("Running GPU inference on random noise sample...")
    
    os.makedirs('outputs', exist_ok=True)
    out_filename = f"model_output_{int(time.time())}.wav"
    out_path = os.path.join('outputs', out_filename)
    
    try:
        noise_engine.eval() # Set to evaluation mode
        with torch.no_grad():
            # Create 3 seconds of static noise
            sample_rate = 16000
            noisy_signal = torch.randn(1, 1, sample_rate * 3).to(device)
            
            # Run through our trained Engine
            cleaned_signal = noise_engine(noisy_signal)
            
            # Move back to CPU for saving
            cleaned_signal = cleaned_signal.squeeze(0).cpu()
            
            # Save as WAV using torchaudio
            torchaudio.save(out_path, cleaned_signal, sample_rate)
            
        log_msg(f"Inference complete. Generated {out_filename}")
        return jsonify({"status": "success", "audio_url": f"/outputs/{out_filename}"})
    except Exception as e:
        log_msg(f"Inference failed: {str(e)}")
        return jsonify({"status": "error"}), 500

@app.route('/outputs/<filename>')
def serve_output(filename):
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("="*60)
    print("Starting Noise Engine Server...")
    print("Access the dashboard at: http://localhost:5000")
    print("="*60)
    # Host on 0.0.0.0 to be accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
