# Audio Stem Separator (DSP Engine)

A digital signal processing (DSP) pipeline designed to isolate and extract distinct musical components (vocals, drums, bass, and accompaniment) from mixed audio recordings. This project implements classical signal processing techniques and time-frequency masking alongside neural separation backends to produce clean stem tracks with minimal artifacts.

---

## Features

- **Multi-Source Demixing:** Separate mixed tracks into 4 core stems: `vocals`, `drums`, `bass`, and `other`.
- **Custom DSP Pre/Post-Processing:**
  - Short-Time Fourier Transform (STFT) with configurable windowing (Hann, Blackman).
  - Magnitude/Phase masking and spectrogram filtering.
  - Phase reconstruction using the Griffin-Lim algorithm and soft-mask synthesis.
  - Artifact reduction and high-frequency spectral interpolation.
- **Batch Processing:** Process entire directories of audio files in single passes.
- **Format Support:** Reads and exports uncompressed (`WAV`, `FLAC`) and compressed formats (`MP3`, `OGG`).
- **Low-Latency Mode:** Streamlined pipeline for near-real-time audio monitoring.

---
