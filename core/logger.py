import logging
import os
from datetime import datetime

def setup_logger():
    # Ruta absoluta al directorio raíz del paquete (music_maker)
    base_dir = os.path.dirname(os.path.abspath(__file__))            # .../music_maker/core
    project_root = os.path.abspath(os.path.join(base_dir, ".."))     # .../music_maker
    logs_dir = os.path.join(project_root, "logs")                    # .../music_maker/logs
    os.makedirs(logs_dir, exist_ok=True)

    log_filename = os.path.join(logs_dir, f"session_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger("MusicEngine")
    logger.setLevel(logging.DEBUG)

    # Evitar handlers duplicados si se llama varias veces
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

        fh = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger

logger = setup_logger()
