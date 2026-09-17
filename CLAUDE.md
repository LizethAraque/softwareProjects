# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

# Run the app
python transcriptor_reuniones.py
```

The Gradio UI starts at `http://localhost:7860` and opens in the browser automatically.

## Installing Dependencies

```bash
pip install gradio openai-whisper torch imageio[ffmpeg] imageio-ffmpeg ffmpeg-python
```

No `requirements.txt` exists — the venv is the source of truth for installed packages.

## Architecture

Single-file application: [transcriptor_reuniones.py](transcriptor_reuniones.py) (~200 lines, Python + Gradio).

**Data flow:**
1. User uploads a video file (MP4, MOV, AVI, MKV, WebM, M4V) via Gradio and selects Whisper model size, language, and whether to include timestamps.
2. Video is copied to `~/transcriptor_tmp/` (temp staging).
3. `_extraer_audio_numpy()` invokes the bundled `imageio_ffmpeg` binary via subprocess to extract raw PCM-16 audio, then converts it to a float32 numpy array at 16 kHz (Whisper's required sample rate).
4. The Whisper model is loaded and called directly with the numpy array — this intentionally bypasses any system-level FFmpeg dependency.
5. Transcription segments are assembled into plain text and/or a timestamped version.
6. Output is saved to `~/Documents/Transcripciones/` as `transcripcion_YYYYMMDD_HHMMSS.txt` / `timestamps_YYYYMMDD_HHMMSS.txt`.

**FFmpeg**: Provided by `imageio_ffmpeg` (bundled binary). A local build (`ffmpeg-8.0.1-essentials_build/`) is also present but not used by the code. FFmpeg availability is checked at startup and printed to console (✅/❌).

## Key Design Decisions

- **No system FFmpeg required**: Audio extraction uses `imageio_ffmpeg.get_ffmpeg_exe()` to locate the bundled binary, avoiding environment dependency issues on Windows.
- **Direct numpy audio path**: Whisper receives a pre-converted float32 array rather than a file path, sidestepping Whisper's internal ffmpeg call.
- **Model is reloaded per transcription**: `whisper.load_model()` is called inside `transcribir()` every time — large model choices will be slow to start.
- **All UI labels and status messages are in Spanish.**
