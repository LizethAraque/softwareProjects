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
    "turbo  – Alta calidad y rápido": "turbo",
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
UMBRAL_SIMILITUD_ORADOR = 0.72
_ENCODER_ORADORES = None


def _formatear_tiempo(segundos):
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _cerrar_parrafo(parrafos, fragmentos, orador):
    if not fragmentos:
        return
    texto = " ".join(fragmentos)
    if orador:
        texto = f"[{orador}] {texto}"
    parrafos.append(texto)


def _debe_cerrar_parrafo(fragmentos, pausa, parrafo_largo, cambio_orador):
    return bool(fragmentos) and (pausa or parrafo_largo or cambio_orador)


def _formatear_transcripcion(segmentos, texto_original):
    if not segmentos:
        return texto_original.strip()

    parrafos = []
    fragmentos = []
    longitud_parrafo = 0
    fin_anterior = None
    ultimo_fragmento = ""
    orador_parrafo = None

    for segmento in segmentos:
        texto = segmento["text"].strip()
        if not texto:
            continue

        inicio = segmento.get("start", 0)
        orador = segmento.get("speaker")
        pausa = fin_anterior is not None and inicio - fin_anterior >= 1.5
        fin_de_oracion = ultimo_fragmento.endswith((".", "!", "?"))
        parrafo_largo = longitud_parrafo >= 400 and fin_de_oracion
        cambio_orador = fragmentos and orador != orador_parrafo

        if _debe_cerrar_parrafo(
            fragmentos, pausa, parrafo_largo, cambio_orador
        ):
            _cerrar_parrafo(parrafos, fragmentos, orador_parrafo)
            fragmentos = []
            longitud_parrafo = 0

        if not fragmentos:
            orador_parrafo = orador
        fragmentos.append(texto)
        longitud_parrafo += len(texto) + (1 if len(fragmentos) > 1 else 0)
        ultimo_fragmento = texto
        fin_anterior = segmento.get("end", inicio)

    _cerrar_parrafo(parrafos, fragmentos, orador_parrafo)

    return "\n\n".join(parrafos)


def _valor_vacio(valor):
    if valor is None:
        return True
    if isinstance(valor, (float, np.floating)) and np.isnan(valor):
        return True
    return not str(valor).strip()


def _parsear_tiempo_muestra(valor):
    texto = str(valor).strip()
    try:
        if ":" in texto:
            partes = texto.split(":")
            if len(partes) != 2:
                raise ValueError
            minutos = int(partes[0])
            segundos = float(partes[1])
            if minutos < 0 or not 0 <= segundos < 60:
                raise ValueError
            return minutos * 60 + segundos
        segundos = float(texto)
        if segundos < 0:
            raise ValueError
        return segundos
    except ValueError as exc:
        raise ValueError(
            f"Tiempo inválido: '{texto}'. Usa mm:ss o segundos."
        ) from exc


def _parsear_fila_orador(fila, numero_fila, duracion_audio):
    if not isinstance(fila, (list, tuple)):
        return None
    valores = list(fila[:3])
    valores.extend([None] * (3 - len(valores)))
    if all(_valor_vacio(valor) for valor in valores):
        return None

    nombre, inicio, fin = valores
    if _valor_vacio(nombre) or _valor_vacio(inicio) or _valor_vacio(fin):
        raise ValueError(
            f"Completa nombre, inicio y fin en la fila {numero_fila}."
        )

    nombre = str(nombre).strip()
    inicio = _parsear_tiempo_muestra(inicio)
    fin = _parsear_tiempo_muestra(fin)
    if fin <= inicio:
        raise ValueError(
            f"El fin de la muestra de {nombre} debe ser posterior al inicio."
        )
    if fin - inicio < 3:
        raise ValueError(
            f"La muestra de {nombre} debe durar al menos 3 segundos; "
            "se recomiendan 5 a 15 segundos claros."
        )
    if fin > duracion_audio:
        raise ValueError(
            f"La muestra de {nombre} termina después de la duración del video."
        )
    return nombre, inicio, fin


def _leer_muestras_oradores(datos, duracion_audio):
    if datos is None:
        return []
    if isinstance(datos, np.ndarray):
        datos = datos.tolist()
    return [
        muestra
        for numero_fila, fila in enumerate(datos, start=1)
        if (muestra := _parsear_fila_orador(
            fila, numero_fila, duracion_audio
        )) is not None
    ]


def _normalizar_embedding(embedding):
    norma = np.linalg.norm(embedding)
    if norma == 0:
        raise ValueError("No se pudo calcular una referencia de voz válida.")
    return embedding / norma


def _obtener_encoder_oradores():
    global _ENCODER_ORADORES
    if _ENCODER_ORADORES is None:
        from resemblyzer import VoiceEncoder

        _ENCODER_ORADORES = VoiceEncoder(device="cpu", verbose=False)
    return _ENCODER_ORADORES


def _preparar_audio_para_encoder(audio):
    from resemblyzer.audio import normalize_volume

    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < SAMPLE_RATE or np.max(np.abs(audio)) < 1e-4:
        return None
    return normalize_volume(audio, target_dBFS=-30, increase_only=True)


def _crear_perfiles_oradores(audio, muestras, encoder):
    referencias = {}
    for nombre, inicio, fin in muestras:
        inicio_muestra = int(inicio * SAMPLE_RATE)
        fin_muestra = int(fin * SAMPLE_RATE)
        audio_muestra = _preparar_audio_para_encoder(
            audio[inicio_muestra:fin_muestra]
        )
        if audio_muestra is None:
            raise ValueError(
                f"No hay suficiente voz en la muestra de {nombre}. "
                "Elige un intervalo más claro."
            )
        embedding = _normalizar_embedding(
            encoder.embed_utterance(audio_muestra)
        )
        referencias.setdefault(nombre, []).append(embedding)

    perfiles = {}
    for nombre, embeddings in referencias.items():
        perfiles[nombre] = _normalizar_embedding(np.mean(embeddings, axis=0))
    return perfiles


def _embedding_segmento(audio, segmento, encoder):
    duracion = len(audio) / SAMPLE_RATE
    inicio = max(0.0, float(segmento.get("start", 0)))
    fin = min(duracion, float(segmento.get("end", inicio)))
    if fin <= inicio:
        return None

    duracion_minima = min(1.6, duracion)
    if fin - inicio < duracion_minima:
        centro = (inicio + fin) / 2
        inicio = max(0.0, centro - duracion_minima / 2)
        fin = min(duracion, inicio + duracion_minima)
        inicio = max(0.0, fin - duracion_minima)

    audio_segmento = audio[int(inicio * SAMPLE_RATE):int(fin * SAMPLE_RATE)]
    audio_limpio = _preparar_audio_para_encoder(audio_segmento)
    if audio_limpio is None:
        return None
    return _normalizar_embedding(encoder.embed_utterance(audio_limpio))


def _mejor_perfil(embedding, perfiles):
    if not perfiles:
        return None, -1.0
    nombre = max(
        perfiles,
        key=lambda key: float(np.dot(embedding, perfiles[key])),
    )
    return nombre, float(np.dot(embedding, perfiles[nombre]))


def _asignar_etiqueta_orador(
    embedding,
    perfiles_conocidos,
    perfiles_nuevos,
    cantidades_nuevas,
    siguiente_orador,
):
    nombre, similitud = _mejor_perfil(embedding, perfiles_conocidos)
    if similitud >= UMBRAL_SIMILITUD_ORADOR:
        return nombre, siguiente_orador

    nombre, similitud = _mejor_perfil(embedding, perfiles_nuevos)
    if similitud < UMBRAL_SIMILITUD_ORADOR:
        while (
            f"Orador {siguiente_orador}" in perfiles_conocidos
            or f"Orador {siguiente_orador}" in perfiles_nuevos
        ):
            siguiente_orador += 1
        nombre = f"Orador {siguiente_orador}"
        siguiente_orador += 1
        perfiles_nuevos[nombre] = embedding
        cantidades_nuevas[nombre] = 0

    cantidad = cantidades_nuevas[nombre]
    perfiles_nuevos[nombre] = _normalizar_embedding(
        perfiles_nuevos[nombre] * cantidad + embedding
    )
    cantidades_nuevas[nombre] = cantidad + 1
    return nombre, siguiente_orador


def _etiquetar_oradores(audio, segmentos, muestras, progress):
    encoder = _obtener_encoder_oradores()
    perfiles_conocidos = _crear_perfiles_oradores(audio, muestras, encoder)
    perfiles_nuevos = {}
    cantidades_nuevas = {}
    siguiente_orador = 1

    for indice, segmento in enumerate(segmentos):
        progress(
            (indice + 1) / max(len(segmentos), 1),
            desc=f"Identificando oradores ({indice + 1}/{len(segmentos)})...",
        )
        embedding = _embedding_segmento(audio, segmento, encoder)
        if embedding is None:
            segmento["speaker"] = "Orador no identificado"
            continue
        segmento["speaker"], siguiente_orador = _asignar_etiqueta_orador(
            embedding,
            perfiles_conocidos,
            perfiles_nuevos,
            cantidades_nuevas,
            siguiente_orador,
        )

    return segmentos


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


def _validar_archivo_video(archivo):
    if archivo is None:
        raise ValueError("Sube un archivo de video antes de transcribir.")
    if FFMPEG_EXE is None:
        raise ValueError("imageio-ffmpeg no está disponible en este entorno.")

    ruta = archivo if isinstance(archivo, str) else getattr(
        archivo, "name", str(archivo)
    )
    if not os.path.exists(ruta):
        raise ValueError(f"Archivo no encontrado: {ruta}")
    return ruta


def _transcribir_audio(
    audio, nombre_modelo, idioma_seleccionado, identificar_oradores,
    muestras, progress,
):
    progress(0, desc="Cargando el modelo...")
    modelo = whisper.load_model(MODELOS[nombre_modelo])
    opciones = {"verbose": False}
    idioma = IDIOMAS[idioma_seleccionado]
    if idioma:
        opciones["language"] = idioma

    progress(0, desc="Transcribiendo...")
    resultado = modelo.transcribe(audio, **opciones)
    segmentos = resultado.get("segments", [])
    if identificar_oradores and segmentos:
        progress(0, desc="Preparando perfiles de voz...")
        resultado["segments"] = _etiquetar_oradores(
            audio, segmentos, muestras, progress
        )
    return resultado


def _formatear_resultados(resultado, incluir_timestamps, identificar_oradores):
    segmentos = resultado.get("segments", [])
    texto = _formatear_transcripcion(segmentos, resultado["text"])
    texto_timestamps = ""

    if incluir_timestamps and segmentos:
        lineas = []
        for segmento in segmentos:
            inicio = _formatear_tiempo(segmento["start"])
            fin = _formatear_tiempo(segmento["end"])
            orador = segmento.get("speaker")
            etiqueta = f" [{orador}]" if orador else ""
            lineas.append(
                f"[{inicio} → {fin}]{etiqueta}\n"
                f"{segmento['text'].strip()}\n"
            )
        texto_timestamps = "\n".join(lineas)

    duracion = segmentos[-1]["end"] if segmentos else 0
    info = (
        f"✅ Transcripción completada\n"
        f"🌐 Idioma    : {resultado.get('language', '?').upper()}\n"
        f"⏱️  Duración  : {_formatear_tiempo(duracion)}\n"
        f"📝 Palabras  : {len(texto.split())}"
    )
    if identificar_oradores and segmentos:
        oradores = {
            segmento["speaker"]
            for segmento in segmentos
            if segmento.get("speaker")
        }
        info += f"\n🗣️ Oradores: {len(oradores)} (revisa las etiquetas)"
    return info, texto, texto_timestamps


def transcribir(
    archivo,
    nombre_modelo,
    idioma_seleccionado,
    incluir_timestamps,
    identificar_oradores,
    muestras_oradores,
    progress=gr.Progress(track_tqdm=True),
):
    ruta_copia = None
    try:
        progress(0, desc="Validando el video...")
        ruta_original = _validar_archivo_video(archivo)

        carpeta_tmp = os.path.join(os.path.expanduser("~"), "transcriptor_tmp")
        os.makedirs(carpeta_tmp, exist_ok=True)
        extension = os.path.splitext(ruta_original)[1] or ".mp4"
        ruta_copia = os.path.join(carpeta_tmp, f"video_temp{extension}")
        shutil.copy2(ruta_original, ruta_copia)

        progress(0, desc="Extrayendo audio...")
        audio = _extraer_audio_numpy(ruta_copia)
        muestras = []
        if identificar_oradores:
            muestras = _leer_muestras_oradores(
                muestras_oradores, len(audio) / SAMPLE_RATE
            )

        resultado = _transcribir_audio(
            audio,
            nombre_modelo,
            idioma_seleccionado,
            identificar_oradores,
            muestras,
            progress,
        )
        salidas = _formatear_resultados(
            resultado, incluir_timestamps, identificar_oradores
        )
        progress(1, desc="Transcripción completada")
        return salidas

    except ValueError as e:
        return f"⚠️ {e}", "", ""
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
            identificar_oradores = gr.Checkbox(
                value=True,
                label="🗣️ Identificar oradores (experimental)",
            )
            muestras_oradores = gr.Dataframe(
                headers=["Nombre o rol", "Inicio (mm:ss)", "Fin (mm:ss)"],
                datatype=["str", "str", "str"],
                value=[["", "", ""] for _ in range(6)],
                row_count=6,
                column_count=3,
                type="array",
                label="Muestras de voz conocidas (opcional)",
                interactive=True,
            )
            gr.Markdown(
                "Una fila por persona. Elige 5–15 s de voz clara, "
                "sin voces superpuestas. Las voces sin muestra se "
                "agrupan como Orador N; revisa la etiqueta."
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
                    descarga = gr.File(
                        label="Descargar transcripción", interactive=False
                    )
                with gr.Tab("🕐 Con timestamps"):
                    ts_out      = gr.Textbox(label="Con marcas de tiempo", lines=22, placeholder="Aquí aparecerán los timestamps…")
                    btn_gts     = gr.Button("💾 Guardar timestamps en Documentos")
                    descarga_ts = gr.File(
                        label="Descargar timestamps", interactive=False
                    )

    btn.click(
        fn=transcribir,
        inputs=[
            archivo_input,
            modelo_selector,
            idioma_selector,
            incluir_ts,
            identificar_oradores,
            muestras_oradores,
        ],
        outputs=[estado, txt_out, ts_out],
    )

    btn_guardar.click(
        fn=lambda t: guardar_transcripcion(t, "transcripcion"),
        inputs=[txt_out],
        outputs=[descarga],
    )

    btn_gts.click(
        fn=lambda t: guardar_transcripcion(t, "timestamps"),
        inputs=[ts_out],
        outputs=[descarga_ts],
    )

    gr.Markdown("---\n🔒 **Todo se procesa localmente.** Los .txt se guardan en `Documentos\\Transcripciones\\`")


if __name__ == "__main__":
    print("=" * 55)
    print("  🎙️  Transcriptor de Reuniones")
    print("=" * 55)
    app.queue().launch(server_name="0.0.0.0", inbrowser=True)