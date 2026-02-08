import numpy as np

class Sequencer:
    def __init__(self, sr=44100):
        self.sr = sr

    def create_track(self, pattern, sound_func, step_dur, expected_len):
        track = np.zeros(expected_len)
        samples_per_step = int(self.sr * step_dur)
        
        for i in range(min(len(pattern), 16)):
            if pattern[i].upper() == "X":
                sound = sound_func()
                start = i * samples_per_step
                actual_end = min(start + len(sound), expected_len)
                track[start:actual_end] += sound[:actual_end - start]
        return track

    def get_default_patterns(self, style):
        # Dubstep y Brostep usan Half-Time (Bombo en 1, Snare en 9)
        if style in ["dubstep", "brostep"]:
            return {
                'k': "X.......X.......", # Kick espaciado
                's': "........X.......", # Snare pesado en el 3er tiempo
                'h': "X.X.X.X.X.X.X.X.", 
                'b': "X...X...X...X..."  # Sincronía para el wobble
            }
        # Mantener los estilos anteriores
        return {'k': "X...X...X...X...", 's': "....X.......X...", 'h': "XXXXXXXXXXXXX.X."}