"""
CORE: MAIN v2.1 - ENTRYPOINT, RENDER ORCHESTRATION, SAFE IO AND GUI AUTO-LAUNCH
- Intenta arrancar una GUI si existe (módulos comunes: ui.gui, gui, core.gui)
- Mantiene modo headless si no hay GUI disponible
- Resto de funcionalidades idénticas a v2.0
"""
import os
import sys
import time
import math
import wave
import traceback
import argparse
from datetime import datetime

import numpy as np

# Asegurar que la raíz del proyecto (music_maker) esté en sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))      # .../music_maker/core
project_root = os.path.abspath(os.path.join(base_dir, ".."))  # .../music_maker
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import logger (asegúrate de que core/logger.py ya está en el proyecto)
try:
    from core.logger import logger
except Exception:
    # Fallback simple logger si core.logger falla
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("MusicEngine")
    logger.warning("No se pudo importar core.logger; usando logger básico.")

# Importar módulos principales del motor
try:
    from compositor.arranger import Arranger
    from engine.effects import Ef
    from engine.oscillators import Osc
    from engine.percussion import Perc
except Exception as e:
    logger.exception("Error importando módulos del motor: %s", e)
    # No abortar inmediatamente; se capturará más abajo si se intenta renderizar.

# -----------------------
# GUI auto-launch helper
# -----------------------
def try_launch_gui(renderer):
    """
    Intentar arrancar la GUI si existe un módulo GUI en el proyecto.
    Busca en varios lugares comunes y llama a run_gui(renderer) o start(renderer).
    No falla si no hay GUI; registra errores si la importación lanza excepciones.
    """
    gui_candidates = [
        ("ui.gui", "run_gui"),
        ("gui", "run_gui"),
        ("core.gui", "run_gui"),
        ("ui.app", "run_gui"),
        ("gui.app", "run_gui"),
        ("ui.gui", "start"),
        ("gui", "start"),
        ("core.gui", "start"),
    ]

    for module_name, func_name in gui_candidates:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name, None)
            if callable(func):
                logger.info("Iniciando GUI desde %s.%s", module_name, func_name)
                try:
                    # Llamada segura: algunos GUIs esperan el renderer o no
                    try:
                        func(renderer)
                    except TypeError:
                        func()
                    return True
                except Exception as e:
                    logger.exception("Error al ejecutar la GUI (%s.%s): %s", module_name, func_name, e)
                    return False
        except ModuleNotFoundError:
            continue
        except Exception as e:
            logger.exception("Error importando módulo GUI %s: %s", module_name, e)
            continue

    logger.info("No se encontró módulo GUI; continuando en modo headless.")
    return False

# -----------------------
# Utilidades de audio
# -----------------------
def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convierte array float32 (-1..1) a int16."""
    if audio.size == 0:
        return np.array([], dtype=np.int16)
    clipped = np.clip(audio, -1.0, 1.0)
    ints = (clipped * 32767.0).astype(np.int16)
    return ints

def save_wav_mono(path: str, audio: np.ndarray, sr: int = 44100):
    """Guarda un buffer mono float32 como WAV 16-bit."""
    if audio.size == 0:
        logger.warning("save_wav_mono: buffer vacío, creando WAV silencioso.")
    ints = float_to_int16(audio)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(sr)
        wf.writeframes(ints.tobytes())
    logger.info("WAV guardado: %s (duración %.2f s)", path, len(audio) / sr if sr else 0.0)

def save_wav_stereo(path: str, left: np.ndarray, right: np.ndarray, sr: int = 44100):
    """Guarda dos buffers mono como WAV estéreo (16-bit)."""
    if left.size != right.size:
        # recortar o tilear para igualar longitudes
        n = min(len(left), len(right))
        left = left[:n]
        right = right[:n]
    interleaved = np.empty((left.size + right.size,), dtype=np.int16)
    l_int = float_to_int16(left)
    r_int = float_to_int16(right)
    interleaved[0::2] = l_int
    interleaved[1::2] = r_int
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())
    logger.info("WAV estéreo guardado: %s (duración %.2f s)", path, len(left) / sr if sr else 0.0)

# -----------------------
# Render orchestration
# -----------------------
class Renderer:
    def __init__(self, sr=44100, bpm=128, renders_dir="renders"):
        self.sr = sr
        self.bpm = bpm
        self.renders_dir = renders_dir
        os.makedirs(self.renders_dir, exist_ok=True)
        self.arranger = None

    def validate_environment(self):
        """Comprueba que los módulos principales están disponibles."""
        missing = []
        if 'Arranger' not in globals():
            missing.append('Arranger')
        if 'Ef' not in globals():
            missing.append('Ef')
        if 'Osc' not in globals():
            missing.append('Osc')
        if 'Perc' not in globals():
            missing.append('Perc')
        if missing:
            logger.error("Dependencias faltantes: %s", ", ".join(missing))
            raise ImportError(f"Dependencias faltantes: {', '.join(missing)}")
        # Instanciar Arranger si no existe
        try:
            self.arranger = Arranger(sr=self.sr)
            self.arranger.bpm = self.bpm
        except Exception as e:
            logger.exception("No se pudo instanciar Arranger: %s", e)
            raise

    def render(self, duration_sec=30, style="techno", emotion="neutral",
               warmth=0.2, stereo_width=1.0, sidechain=True, filename=None):
        """
        Renderiza la composición completa y devuelve el buffer mono float32.
        - duration_sec: duración en segundos
        - style: string (techno, trance, dubstep, idm...)
        - emotion: etiqueta para Scales
        - warmth: 0..1 (aplicado en master)
        - stereo_width: >1 para ensanchar
        - sidechain: habilita sidechain (ya manejado por Arranger)
        - filename: si se pasa, guarda el WAV en ese path relativo a renders_dir
        """
        start_time = time.time()
        logger.info("Render iniciado: style=%s, emotion=%s, duration=%.1fs", style, emotion, duration_sec)
        try:
            if self.arranger is None:
                self.validate_environment()

            # Componer
            audio = self.arranger.compose(style=style, duration_sec=duration_sec, emotion=emotion)
            if audio is None:
                raise RuntimeError("Arranger devolvió None en lugar de buffer de audio.")

            # Post procesamiento maestro
            audio = Ef.tube_warmth(audio, drive=1.0 + (warmth * 0.5), warmth=warmth)
            audio = Ef.master_limiter(audio, ceiling=0.98)

            # Guardado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = filename or f"{style}_{emotion}_{duration_sec}s_{timestamp}.wav"
            out_path = os.path.join(self.renders_dir, safe_name)

            # Si stereo_width > 1, generar pseudo-estéreo
            if stereo_width and stereo_width > 1.0:
                stereo = Ef.stereo_widener(audio, width=stereo_width, sr=self.sr)
                # stereo_widener devuelve array 2xN
                if isinstance(stereo, np.ndarray) and stereo.ndim == 2 and stereo.shape[0] == 2:
                    save_wav_stereo(out_path, stereo[0], stereo[1], sr=self.sr)
                else:
                    # fallback: save mono
                    save_wav_mono(out_path, audio, sr=self.sr)
            else:
                save_wav_mono(out_path, audio, sr=self.sr)

            elapsed = time.time() - start_time
            logger.info("Render completado en %.2f s. Archivo: %s", elapsed, out_path)
            return out_path, audio

        except Exception as e:
            logger.error("Error durante render: %s", e)
            tb = traceback.format_exc()
            logger.debug("Stack trace:\n%s", tb)
            raise

    def render_short_preview(self, duration_sec=10, **kwargs):
        """Conveniencia para pruebas rápidas."""
        logger.info("Render de prueba (preview) solicitado: %ds", duration_sec)
        return self.render(duration_sec=duration_sec, **kwargs)

# -----------------------
# CLI / Entrypoint
# -----------------------
def parse_args(argv):
    p = argparse.ArgumentParser(prog="music_maker", description="Renderizador Music Maker (headless).")
    p.add_argument("--duration", "-d", type=float, default=30.0, help="Duración en segundos")
    p.add_argument("--style", "-s", type=str, default="techno", help="Estilo (techno, trance, dubstep, idm)")
    p.add_argument("--emotion", "-e", type=str, default="cyberpunk", help="Etiqueta emocional para progresiones")
    p.add_argument("--bpm", type=int, default=128, help="BPM del proyecto")
    p.add_argument("--warmth", type=float, default=0.18, help="Calor analógico 0..1")
    p.add_argument("--stereo", type=float, default=1.0, help="Ancho estéreo (>1 ensancha)")
    p.add_argument("--no-sidechain", dest="sidechain", action="store_false", help="Deshabilitar sidechain")
    p.add_argument("--preview", action="store_true", help="Render corto de prueba (10s)")
    p.add_argument("--out", type=str, default=None, help="Nombre de archivo de salida (opcional)")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    logger.info("Parámetros: %s", vars(args))

    renderer = Renderer(sr=44100, bpm=args.bpm, renders_dir="renders")
    try:
        renderer.validate_environment()
    except Exception as e:
        logger.error("Validación fallida: %s", e)
        print("Error: revisar logs para más detalles.")
        return 1

    # Intentar lanzar GUI si existe; si arranca, la GUI toma el control y salimos.
    try:
        launched = try_launch_gui(renderer)
        if launched:
            logger.info("GUI iniciada; main cede control a la interfaz.")
            return 0
    except Exception:
        logger.exception("Fallo al intentar lanzar la GUI; continuando en modo headless.")

    try:
        if args.preview:
            out, _ = renderer.render_short_preview(duration_sec=10, style=args.style,
                                                   emotion=args.emotion, warmth=args.warmth,
                                                   stereo_width=args.stereo, sidechain=args.sidechain,
                                                   filename=args.out)
        else:
            out, _ = renderer.render(duration_sec=args.duration, style=args.style,
                                     emotion=args.emotion, warmth=args.warmth,
                                     stereo_width=args.stereo, sidechain=args.sidechain,
                                     filename=args.out)
        print(f"Render completado: {out}")
        return 0
    except Exception as e:
        logger.exception("Fallo en main render: %s", e)
        print("Render fallido. Revisa los logs para más información.")
        return 2

if __name__ == "__main__":
    sys.exit(main())
