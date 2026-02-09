import logging
import os
from datetime import datetime

def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    log_filename = os.path.join('logs', f"session_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler() # También lo verás en la consola
        ]
    )
    return logging.getLogger("MusicEngine")

logger = setup_logger()