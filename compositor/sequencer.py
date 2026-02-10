"""
SEQUENCER v12.2 - MERGED ALGORITHMIC RHYTHM ENGINE
Añadido: polirritmia de hi-hat, micro-swing por paso y utilidades para patrones euclidianos.
"""
import numpy as np
import random
from typing import Iterable, Union

class Sequencer:
    """
    ENGINE: SEQUENCER v12.2 - ALGORITHMIC RHYTHM ENGINE
    Maneja la estructura temporal, groove, polirritmia, swing y probabilidad.
    """
    def __init__(self, sr=44100, bpm=128):
        self.sr = sr
        self.bpm = bpm

    def create_track(self, pattern: Union[str, Iterable], sound_func, step_dur: float,
                     expected_len: int, swing: float = 0.0, probability: float = 1.0,
                     velocity_map: dict = None):
        """
        Crea una pista con soporte para Swing, Probabilidad y Velocidad.
        - pattern: iterable de caracteres ('X','x','.', etc.) o lista booleana.
        - sound_func: función que genera el audio (acepta sr=... o no).
        - step_dur: duración de cada paso en segundos.
        - expected_len: longitud en muestras del buffer destino.
        - swing: 0.0 a 0.5 (retraso de notas pares).
        - probability: 0.0 a 1.0 (chance de que el paso suene).
        - velocity_map: dict {step_index: velocity}
        """
        track = np.zeros(expected_len, dtype=np.float32)
        samples_per_step = int(self.sr * step_dur)

        # Normalize pattern to list for indexing
        if isinstance(pattern, str):
            pattern_iter = list(pattern)
        else:
            pattern_iter = list(pattern)

        for i, char in enumerate(pattern_iter):
            # Probabilidad algorítmica
            if random.random() > probability:
                continue

            # Interpretación de patrón: 'X' = golpe, '.' = silencio
            is_hit = False
            if isinstance(char, str):
                is_hit = (char.upper() == "X")
            else:
                is_hit = bool(char)

            if not is_hit:
                continue

            # Swing: retraso en pasos impares
            swing_offset = 0
            if i % 2 != 0 and swing:
                # micro-swing: añade una pequeña variación aleatoria alrededor del swing
                micro = int(samples_per_step * (swing * (0.8 + 0.4 * random.random())))
                swing_offset = micro

            start = (i * samples_per_step) + swing_offset

            # Dinámica / velocity
            velocity = 1.0 if (isinstance(char, str) and char.isupper()) else 0.5
            if velocity_map and i in velocity_map:
                velocity = velocity_map[i]

            # Generar sonido (intentar pasar sr si la función lo acepta)
            try:
                sample = sound_func(sr=self.sr)
            except TypeError:
                sample = sound_func()

            # Inserción segura en el buffer destino
            actual_end = min(start + len(sample), expected_len)
            if start < expected_len:
                track[start:actual_end] += sample[:actual_end - start] * velocity

        return track

    # --- MOTOR EUCLIDIANO (Bjorklund robusto) ---
    @staticmethod
    def generate_euclidean(steps: int, pulses: int) -> str:
        """
        Distribuye 'pulses' en 'steps' de forma equitativa.
        Retorna un string con 'X' para pulso y '.' para silencio.
        Implementación robusta del algoritmo de Bjorklund.
        """
        if pulses <= 0 or steps <= 0:
            return "." * steps
        if pulses >= steps:
            return "X" * steps

        pattern = []
        counts = []
        remainders = []
        divisor = steps - pulses
        remainders.append(pulses)
        level = 0
        while True:
            counts.append(divisor // remainders[level])
            remainders.append(divisor % remainders[level])
            level += 1
            if remainders[level] <= 1:
                break
        counts.append(remainders[level])

        def build(l):
            if l == -1:
                return [0]
            if l == -2:
                return [1]
            seq = []
            a = build(l - 1)
            b = build(l - 2)
            seq.extend(a * counts[l])
            seq.extend(b)
            return seq

        seq = build(level)
        seq = seq[:steps]
        return "".join('X' if x == 1 else '.' for x in seq)

    # --- POLYRHYTHM HI-HAT GENERATOR ---
    def generate_polyrhythm_hihat(self, steps_main: int = 16, steps_poly: int = 12,
                                  pulses_poly: int = 4, accent_map: dict = None):
        """
        Genera un patrón de hi-hat polirrítmico mezclando:
        - un patrón base de 'steps_main' (por ejemplo 16)
        - un patrón polirrítmico de 'steps_poly' (por ejemplo 12)
        Devuelve un string de longitud 'steps_main' con 'X' y '.'.
        accent_map: dict {index: velocity_multiplier} para acentos.
        """
        base = ['.' for _ in range(steps_main)]
        poly = self.generate_euclidean(steps_poly, pulses_poly)
        # Map poly pattern onto main grid by stretching/repeating
        for i, ch in enumerate(poly):
            if ch == 'X':
                # position in main grid (round to nearest)
                pos = int(round(i * (steps_main / steps_poly))) % steps_main
                base[pos] = 'X' if base[pos] == '.' else 'X'  # ensure hit
        # Optionally add subtle ghost hits to fill gaps
        for i in range(steps_main):
            if base[i] == '.' and random.random() < 0.06:
                base[i] = 'x'  # ghost note (lower velocity)
        return "".join(base)

    # --- BIBLIOTECA DE PATRONES INTELIGENTES ---
    def get_intelligent_patterns(self, style="techno", complexity=0.5):
        """
        Retorna patrones que mutan según la complejidad deseada.
        Combina presets fijos y patrones euclidianos para IDM/Techno.
        """
        patterns = {
            "techno": {
                'k': "X...X...X...X...",
                's': "....X.......X...",
                'h': "X.X.X.X.X.X.X.X.",
                'o': "..X...X...X...X."  # Offbeat hi-hat
            },
            "idm": {
                'k': self.generate_euclidean(16, random.randint(3, 6)),
                's': self.generate_euclidean(16, random.randint(2, 4)),
                'h': "XxxXxxXxxXxxXxxX"
            }
        }

        if style in ["dubstep", "brostep"]:
            patterns[style] = {
                'k': "X...............",
                's': "........X.......",
                'h': "X.X.X.X.X.X.X.X."
            }

        # Ajuste por complejidad: si complexity alto, añadir más hits en 'idm'
        if style == "idm" and complexity > 0.7:
            patterns["idm"]['h'] = "XxXXxXxXXxXxXxXx"

        return patterns.get(style, patterns["techno"])

    # --- UTILIDADES DE CONSTRUCCIÓN Y TIEMPO ---
    def build(self, sections):
        """
        Concatena buffers de secciones en orden.
        sections: iterable de arrays (puede contener arrays vacíos).
        """
        return np.concatenate([s for s in sections if getattr(s, "size", None) and len(s) > 0]) if sections else np.array([], dtype=np.float32)

    def bpm_to_step_dur(self, bpm=None, division=4):
        """
        Convierte BPM a duración de paso en segundos.
        division: subdivisión por negra (4 = semicorchea por defecto).
        """
        b = bpm or self.bpm
        return (60.0 / b) / division

# Alias
Seq = Sequencer
