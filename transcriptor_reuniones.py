"""
Transcriptor de Reuniones
Solución definitiva: extrae el audio manualmente con imageio_ffmpeg
para que Whisper nunca necesite llamar a ffmpeg del sistema.
"""

import gradio as gr
import whisper
import numpy as np
import os
import shutil
import subprocess
import struct
from datetime import datetime


# ── Verificar ffmpeg de imageio al arrancar ───────────────────────────────────
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"✅ ffmpeg listo en: {FFMPEG_EXE}")
except Exception as e:
    FFMPEG_EXE = None
    print(f"❌ imageio_ffmpeg no disponible: {e}")
# ─────────────────────────────────────────────────────────────────────────────


MODELOS = {
    "tiny   – Rápido (menor precisión)": "tiny",
    "base   – Equilibrado (recomendado)": "base",
    "small  – Buena precisión": "small",
    "medium – Alta precisión (más lento)": "medium",
    "large  – Máxima precisión (muy lento)": "large",
}

IDIOMAS = {
    "Detección automática": None,
    "Español": "es",
    "Inglés": "en",
    "Portugués": "pt",
    "Francés": "fr",
    "Alemán": "de",
    "Italiano": "it",
}

SAMPLE_RATE = 16000  # Whisper siempre necesita 16kHz


def _formatear_tiempo(segundos):
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extraer_audio_numpy(ruta_video: str) -> np.ndarray:
    """
    Usa imageio_ffmpeg directamente para extraer el audio del video
    y lo devuelve como numpy array float32 a 16kHz.
    Whisper recibe el array y nunca llama ffmpeg del sistema.
    """
    cmd = [
        FFMPEG_EXE,
        "-i", ruta_video,
        "-f", "s16le",        # PCM 16-bit little-endian
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",           # mono
        "-",                  # salida a stdout
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr.decode(errors='ignore')}")

    raw = result.stdout
    # Convertir bytes PCM s16le a float32
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples


def transcribir(archivo, nombre_modelo, idioma_seleccionado, incluir_timestamps):
    ruta_copia = None
    try:
        if archivo is None:
            return "⚠️ Sube un archivo de video antes de transcribir.", "", ""

        if FFMPEG_EXE is None:
            return "❌ Ejecuta primero:\n  pip install imageio[ffmpeg]\nLuego reinicia el programa.", "", ""

        ruta_original = archivo if isinstance(archivo, str) else getattr(archivo, "name", str(archivo))

        if not os.path.exists(ruta_original):
            return f"❌ Archivo no encontrado:\n{ruta_original}", "", ""

        # Copiar a ruta sin espacios (precaución extra)
        carpeta_tmp = os.path.join(os.path.expanduser("~"), "transcriptor_tmp")
        os.makedirs(carpeta_tmp, exist_ok=True)
        ext = os.path.splitext(ruta_original)[1] or ".mp4"
        ruta_copia = os.path.join(carpeta_tmp, f"video_temp{ext}")
        shutil.copy2(ruta_original, ruta_copia)

        # Extraer audio como numpy array (sin llamar ffmpeg del sistema)
        audio_np = _extraer_audio_numpy(ruta_copia)

        # Cargar modelo Whisper y transcribir pasando el array directamente
        modelo = whisper.load_model(MODELOS[nombre_modelo])
        opciones = {"verbose": False}
        idioma_key = IDIOMAS[idioma_seleccionado]
        if idioma_key:
            opciones["language"] = idioma_key

        # Whisper acepta numpy array float32 directamente
        resultado = modelo.transcribe(audio_np, **opciones)

        texto_plano = resultado["text"].strip()

        texto_timestamps = ""
        if incluir_timestamps and resultado.get("segments"):
            lineas = []
            for seg in resultado["segments"]:
                t0 = _formatear_tiempo(seg["start"])
                t1 = _formatear_tiempo(seg["end"])
                lineas.append(f"[{t0} → {t1}]\n{seg['text'].strip()}\n")
            texto_timestamps = "\n".join(lineas)

        idioma_det = resultado.get("language", "?").upper()
        duracion   = resultado["segments"][-1]["end"] if resultado["segments"] else 0
        info = (
            f"✅ Transcripción completada\n"
            f"🌐 Idioma    : {idioma_det}\n"
            f"⏱️  Duración  : {_formatear_tiempo(duracion)}\n"
            f"📝 Palabras  : {len(texto_plano.split())}"
        )
        return info, texto_plano, texto_timestamps

    except Exception as e:
        import traceback
        return f"❌ Error:\n{str(e)}\n\n{traceback.format_exc()}", "", ""
    finally:
        if ruta_copia and os.path.exists(ruta_copia):
            try:
                os.remove(ruta_copia)
            except Exception:
                pass


def guardar_transcripcion(texto, prefijo="transcripcion"):
    if not texto or not texto.strip():
        return None
    carpeta = os.path.join(os.path.expanduser("~"), "Documents", "Transcripciones")
    os.makedirs(carpeta, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta, f"{prefijo}_{timestamp}.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)
    return ruta


with gr.Blocks(title="Transcriptor de Reuniones") as app:
    gr.Markdown("# 🎙️ Transcriptor de Reuniones\nSube tu video y obtén la transcripción de forma local y gratuita.")

    with gr.Row():
        with gr.Column(scale=1):
            archivo_input = gr.File(
                label="📹 Sube tu video (MP4, MOV, AVI, MKV…)",
                file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"],
            )
            modelo_selector = gr.Dropdown(choices=list(MODELOS.keys()), value="base   – Equilibrado (recomendado)", label="🤖 Modelo")
            idioma_selector = gr.Dropdown(choices=list(IDIOMAS.keys()), value="Español", label="🌐 Idioma")
            incluir_ts      = gr.Checkbox(value=True, label="⏱️ Incluir timestamps")
            btn             = gr.Button("🚀 Iniciar Transcripción", variant="primary", size="lg")
            estado          = gr.Textbox(label="ℹ️ Estado", lines=6, interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📄 Transcripción"):
                    txt_out     = gr.Textbox(label="Texto", lines=22, placeholder="Aquí aparecerá el resultado…")
                    btn_guardar = gr.Button("💾 Guardar .txt en Documentos")
                    descarga    = gr.File(label="Archivo guardado", visible=False)
                with gr.Tab("🕐 Con timestamps"):
                    ts_out      = gr.Textbox(label="Con marcas de tiempo", lines=22, placeholder="Aquí aparecerán los timestamps…")
                    btn_gts     = gr.Button("💾 Guardar timestamps en Documentos")
                    descarga_ts = gr.File(label="Archivo guardado", visible=False)

    btn.click(fn=transcribir, inputs=[archivo_input, modelo_selector, idioma_selector, incluir_ts], outputs=[estado, txt_out, ts_out])

    btn_guardar.click(fn=lambda t: guardar_transcripcion(t, "transcripcion"), inputs=[txt_out], outputs=[descarga]
    ).then(fn=lambda f: gr.File(visible=bool(f), value=f), inputs=[descarga], outputs=[descarga])

    btn_gts.click(fn=lambda t: guardar_transcripcion(t, "timestamps"), inputs=[ts_out], outputs=[descarga_ts]
    ).then(fn=lambda f: gr.File(visible=bool(f), value=f), inputs=[descarga_ts], outputs=[descarga_ts])

    gr.Markdown("---\n🔒 **Todo se procesa localmente.** Los .txt se guardan en `Documentos\\Transcripciones\\`")


if __name__ == "__main__":
    print("=" * 55)
    print("  🎙️  Transcriptor de Reuniones")
    print("=" * 55)
    app.launch(server_name="0.0.0.0", inbrowser=True)