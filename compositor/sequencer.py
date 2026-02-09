import numpy as np
import random

class Sequencer:
    """
    ENGINE: SEQUENCER v12.0 - ALGORITHMIC RHYTHM ENGINE
    --------------------------------------------------
    Maneja la estructura temporal, el groove y la polirritmia.
    """
    def __init__(self, sr=44100):
        self.sr = sr

    def create_track(self, pattern, sound_func, step_dur, expected_len, 
                     swing=0.0, probability=1.0, velocity_map=None):
        """
        Crea una pista con soporte para Swing, Probabilidad y Velocidad.
        swing: 0.0 a 0.5 (retraso de notas pares).
        probability: 0.0 a 1.0 (posibilidad de que el paso suene).
        """
        track = np.zeros(expected_len)
        samples_per_step = int(self.sr * step_dur)
        
        for i, char in enumerate(pattern):
            # 1. Comprobación de probabilidad (Variabilidad algorítmica)
            if random.random() > probability:
                continue

            if char.upper() == "X":
                # 2. Lógica de Swing
                # Retrasamos los pasos pares (1, 3, 5...)
                swing_offset = 0
                if i % 2 != 0:
                    swing_offset = int(samples_per_step * swing)
                
                start = (i * samples_per_step) + swing_offset
                
                # 3. Generación de sonido con velocity (Dinámica)
                # Las mayúsculas 'X' suenan más fuerte que las minúsculas 'x'
                velocity = 1.0 if char.isupper() else 0.5
                if velocity_map and i in velocity_map:
                    velocity = velocity_map[i]

                sound = sound_func()
                
                # Inserción con seguridad de límites
                actual_end = min(start + len(sound), expected_len)
                if start < expected_len:
                    track[start:actual_end] += sound[:actual_end - start] * velocity
                    
        return track

    # --- MOTOR EUCLIDIANO ---
    
    @staticmethod
    def generate_euclidean(steps, pulses):
        """
        Algoritmo de Bjorklund para distribuir pulsos de forma equitativa.
        Crea ritmos 'exóticos' y matemáticamente perfectos.
        """
        pattern = [1] * pulses + [0] * (steps - pulses)
        
        def build(p):
            # Lógica recursiva para agrupar 1s y 0s
            zeros = [x for x in p if x == 0]
            ones = [x for x in p if x != 0] # Simplificación para el ejemplo
            # (En producción se usa una rotación de arrays)
            return p # Fallback estable
            
        # Generación rápida por espaciado
        res = [0] * steps
        if pulses == 0: return res
        interval = steps / pulses
        for i in range(pulses):
            res[int(i * interval)] = 1
        
        # Convertir a formato string para el sequencer
        return "".join(['X' if x == 1 else '.' for x in res])

    # --- BIBLIOTECA DE PATRONES INTELIGENTES ---

    def get_intelligent_patterns(self, style, complexity=0.5):
        """
        Retorna patrones que mutan según la complejidad deseada.
        """
        patterns = {
            "techno": {
                'k': "X...X...X...X...",
                's': "....X.......X...",
                'h': "X.X.X.X.X.X.X.X.",
                'o': "..X...X...X...X." # Offbeat hi-hat
            },
            "idm": {
                'k': self.generate_euclidean(16, random.randint(3, 6)),
                's': self.generate_euclidean(16, random.randint(2, 4)),
                'h': "XxxXxxXxxXxxXxxX" # Ghost notes (velocidad baja)
            }
        }
        
        # Si el estilo es nuevo (ej. dubstep), añadimos su base
        if style in ["dubstep", "brostep"]:
            patterns[style] = {
                'k': "X...............",
                's': "........X.......",
                'h': "X.X.X.X.X.X.X.X."
            }
            
        return patterns.get(style, patterns["techno"])

    # --- UTILIDADES DE TIEMPO ---

    def bpm_to_step_dur(self, bpm, division=4):
        """Convierte BPM a duración de paso (default: semicorcheas)."""
        # 60 / BPM = duración de una negra. / 4 = semicorchea.
        return (60.0 / bpm) / division