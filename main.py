import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, firwin, filtfilt, stft, istft, medfilt2d

# ==============================================================================
# 1. INGESTION, PRE-PROCESSING & SIGNAL PROFILING
# ==============================================================================
FILENAME = 'input_audio.wav'  # Ensure your .wav file is in the same directory

print("[*] Ingesting audio file...")
fs, raw_data = wavfile.read(FILENAME)

# Downmix to mono if stereo
if raw_data.ndim > 1:
    raw_data = raw_data.mean(axis=1)

# Normalize amplitude to dynamic range [-1.0, 1.0]
signal = raw_data.astype(np.float64)
max_val = np.max(np.abs(signal))
if max_val > 0:
    signal = signal / max_val

total_samples = len(signal)
total_duration = total_samples / fs
peak_amp = np.max(np.abs(signal))
rms_val = np.sqrt(np.mean(signal**2))

print("\n" + "=" * 50)
print("             SIGNAL PROFILING METRICS")
print("=" * 50)
print(f"Sampling Frequency (fs) : {fs} Hz")
print(f"Total Duration          : {total_duration:.2f} seconds")
print(f"Total Sample Count      : {total_samples}")
print(f"Peak Normalized Amp     : {peak_amp:.4f}")
print(f"RMS Energy              : {rms_val:.4f}")
print("=" * 50 + "\n")

# Use a 10-second segment for spectral analysis
analysis_dur = min(10.0, total_duration)
x = signal[: int(analysis_dur * fs)]
N = len(x)
t = np.linspace(0, N / fs, N, endpoint=False)

# ==============================================================================
# 2. 2D STFT HARMONIC-PERCUSSIVE SOUND SEPARATION (HPSS)
# ==============================================================================
print("[*] Computing 2D STFT Harmonic-Percussive Separation...")
f_bins, t_frames, Zxx = stft(x, fs=fs, nperseg=2048, noverlap=1024)
magnitude = np.abs(Zxx)

# Directional median filters: Horizontal (Harmonics) vs. Vertical (Percussion)
H_mag = medfilt2d(magnitude, kernel_size=(1, 31))
P_mag = medfilt2d(magnitude, kernel_size=(31, 1))

# Generate soft masks
mask_H = (H_mag ** 2) / (H_mag ** 2 + P_mag ** 2 + 1e-12)
mask_P = (P_mag ** 2) / (H_mag ** 2 + P_mag ** 2 + 1e-12)

# Invert back to time-domain waveforms
_, y_harmonic = istft(mask_H * Zxx, fs=fs)
_, y_drums = istft(mask_P * Zxx, fs=fs)

# Align lengths with original snippet
y_harmonic = y_harmonic[:N]
y_drums = y_drums[:N]

# ==============================================================================
# 3. SUBBAND DIGITAL FILTER DESIGN & ZERO-PHASE FILTERING
# ==============================================================================
print("[*] Designing and applying digital subband filters...")

# Filter A: 6th-Order Butterworth Low-Pass (Bass/Sub: fc = 250 Hz)
b_low, a_low = butter(6, 250.0, btype='low', fs=fs)
y_bass = filtfilt(b_low, a_low, y_harmonic)

# Filter B: 401-Tap Type-I Linear-Phase FIR Band-Pass (Vocals/Mids: 300 - 3400 Hz)
num_taps = 401
b_mid = firwin(num_taps, [300.0, 3400.0], pass_zero='bandpass', fs=fs, window='hamming')
y_vocals = filtfilt(b_mid, [1.0], y_harmonic)

# Filter C: 4th-Order Butterworth High-Pass (Cymbals/Treble: fc = 4500 Hz)
b_high, a_high = butter(4, 4500.0, btype='high', fs=fs)
y_treble = filtfilt(b_high, a_high, x)

# Recombination & Residual Calculation
y_recomb = y_bass + y_vocals + y_treble + y_drums
x_loss = x - y_recomb

# ==============================================================================
# 4. EXPORT EXTRACTED AUDIO COMPONENTS (.WAV)
# ==============================================================================
def export_wav(filename, audio_data, sample_rate):
    norm_audio = audio_data / (np.max(np.abs(audio_data)) + 1e-12)
    int_audio = np.int16(norm_audio * 32767)
    wavfile.write(filename, sample_rate, int_audio)

export_wav('bass_stem.wav', y_bass, fs)
export_wav('vocals_stem.wav', y_vocals, fs)
export_wav('treble_stem.wav', y_treble, fs)
export_wav('drums_stem.wav', y_drums, fs)
export_wav('full_recombination.wav', y_recomb, fs)
print("[+] Audio files successfully exported (.wav): Bass, Vocals, Treble, Drums, Recombined.\n")

# ==============================================================================
# 5. HIGH-RESOLUTION FFT SPECTRAL COMPUTATION
# ==============================================================================
N_fft = 8192
win = np.hanning(min(N, N_fft))

def get_fft(sig):
    sig_seg = sig[:len(win)] * win
    X_f = np.fft.rfft(sig_seg, n=N_fft)
    return 20 * np.log10(np.abs(X_f) + 1e-6)

freqs = np.fft.rfftfreq(N_fft, d=1/fs)
mag_orig = get_fft(x)
mag_bass = get_fft(y_bass)
mag_voc = get_fft(y_vocals)
mag_treb = get_fft(y_treble)
mag_drum = get_fft(y_drums)
mag_rec = get_fft(y_recomb)

# ==============================================================================
# 6. DUAL-DOMAIN COMPONENT VISUALIZATION
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

# --- PLOT 1: TIME-DOMAIN WAVEFORMS (ACTIVE 30 ms WINDOW) ---
# Start from 2.0s to skip any introductory silence or filter edge transients
start_t = 2.0 if total_duration > 3.0 else 0.5
start_idx = int(start_t * fs)
plot_len = int(0.030 * fs)  # 30 ms window
time_ms = np.linspace(0, (plot_len / fs) * 1000, plot_len)
slice_idx = slice(start_idx, start_idx + plot_len)

ax1.plot(time_ms, x[slice_idx], color='gray', alpha=0.5, lw=1.2, label='Original Mix (Composite)')
ax1.plot(time_ms, y_bass[slice_idx], color='#1f77b4', lw=2.0, label='Bass Stem (Butterworth LPF < 250Hz)')
ax1.plot(time_ms, y_vocals[slice_idx], color='#ff7f0e', lw=1.6, label='Vocals Stem (Linear-Phase FIR 300-3400Hz)')
ax1.plot(time_ms, y_treble[slice_idx], color='#9467bd', lw=1.4, alpha=0.85, label='Treble Stem (Butterworth HPF > 4500Hz)')
ax1.plot(time_ms, y_drums[slice_idx], color='#d62728', lw=1.4, alpha=0.8, linestyle=':', label='Drums Stem (2D HPSS Transients)')
ax1.plot(time_ms, y_recomb[slice_idx], color='#2ca02c', lw=1.5, linestyle='--', alpha=0.9, label='Full Recombination')

ax1.set_title(f'Time-Domain Waveforms (Active 30 ms Snapshot at t = {start_t:.1f} s)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Time (ms)', fontsize=11)
ax1.set_ylabel('Normalized Amplitude', fontsize=11)
ax1.set_xlim(0, time_ms[-1])
ax1.set_ylim(-1.05, 1.05)
ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)

# --- PLOT 2: FREQUENCY SPECTRUM (LOG-MAGNITUDE FFT) ---
ax2.plot(freqs, mag_orig, color='gray', alpha=0.45, lw=1.0, label='Original Spectrum')
ax2.plot(freqs, mag_bass, color='#1f77b4', lw=2.0, label='Bass Spectrum (Low Plateau)')
ax2.plot(freqs, mag_voc, color='#ff7f0e', lw=1.8, label='Vocal Spectrum (Formant Corridor)')
ax2.plot(freqs, mag_treb, color='#9467bd', lw=1.6, label='Treble Spectrum (Air & Cymbals)')
ax2.plot(freqs, mag_drum, color='#d62728', lw=1.2, linestyle=':', alpha=0.8, label='Drums Spectrum (Broadband)')
ax2.plot(freqs, mag_rec, color='#2ca02c', lw=1.6, linestyle='--', alpha=0.85, label='Recombined Spectrum')

# Shaded spectral zones for visual component verification
ax2.axvspan(20, 250, color='#1f77b4', alpha=0.08, label='Bass Subband (20-250 Hz)')
ax2.axvspan(300, 3400, color='#ff7f0e', alpha=0.08, label='Vocal Subband (300-3400 Hz)')
ax2.axvspan(4500, fs / 2, color='#9467bd', alpha=0.08, label='Treble Subband (> 4500 Hz)')

ax2.set_xscale('log')
ax2.set_xlim(20, fs / 2)
ax2.set_ylim(-85, np.max(mag_orig) + 8)
ax2.set_title(f'Frequency Spectrum Comparison (N_FFT = {N_fft}, Hann Window, Δf ≈ {fs/N_fft:.2f} Hz)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Frequency (Hz) - Logarithmic Scale', fontsize=11)
ax2.set_ylabel('Magnitude (dB)', fontsize=11)
ax2.legend(loc='lower left', ncol=2, frameon=True, framealpha=0.9, fontsize=9)
ax2.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
