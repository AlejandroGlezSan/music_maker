import numpy as np

class Oscillators:
    @staticmethod
    def wobble_bass(t, freq, lfo_freq):
        """Bajo con modulación de volumen/filtro rítmico."""
        # El LFO controla el "grosor" y volumen
        lfo = (np.sin(2 * np.pi * lfo_freq * t) + 1) / 2
        base_wave = np.sin(2 * np.pi * freq * t) + 0.5 * np.sign(np.sin(2 * np.pi * freq * t))
        return base_wave * lfo

    @staticmethod
    def brostep_growl(t, freq, intensity):
        """Sonido tipo 'monstruo' usando distorsión de fase y armónicos."""
        # Modulación agresiva de la forma de onda
        wave = np.sin(2 * np.pi * freq * t + intensity * np.sin(2 * np.pi * (freq * 0.5) * t))
        # Añadimos un poco de 'ruido' metálico
        noise = np.random.uniform(-0.1, 0.1, len(t)) * intensity
        return np.clip(wave + noise, -0.8, 0.8)

    @staticmethod
    def sub_bass(t, freq):
        return np.sin(2 * np.pi * freq * t)

    @staticmethod
    def supersaw(t, freq, detune=0.01, num_layers=5):
        signal = np.zeros_like(t)
        for i in range(num_layers):
            f_factor = 1.0 + (detune * (i - (num_layers - 1) / 2))
            signal += 2 * (t * (freq * f_factor) - np.floor(t * (freq * f_factor) + 0.5))
        return signal / num_layers

    @staticmethod
    def sine(t, freq):
        return np.sin(2 * np.pi * freq * t)