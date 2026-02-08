import numpy as np

class Percussion:
    @staticmethod
    def kick(sr=44100, style="techno"):
        duration = 0.4 if style == "techno" else 0.25
        t = np.linspace(0, duration, int(sr * duration))
        # Techno = Sweep lento y grave | Trance = Sweep rápido y seco
        f_start = 150 if style == "techno" else 220
        freq_env = f_start * np.exp(-20 * t) + 40
        phase = 2 * np.pi * np.cumsum(freq_env) / sr
        return np.sin(phase) * np.exp(-6 * t)

    @staticmethod
    def snare(sr=44100):
        t = np.linspace(0, 0.2, int(sr * 0.2))
        noise = np.random.uniform(-1, 1, len(t))
        return (np.sin(2 * np.pi * 180 * t) + noise * 0.4) * np.exp(-15 * t)

    @staticmethod
    def hihat(sr=44100):
        t = np.linspace(0, 0.05, int(sr * 0.05))
        return np.random.uniform(-1, 1, len(t)) * np.exp(-30 * t) * 0.2
    
    @staticmethod
    def white_noise_riser(sr, duration):
        """Ruido que sube de intensidad para transiciones."""
        t = np.linspace(0, duration, int(sr * duration))
        noise = np.random.uniform(-1, 1, len(t))
        # El volumen sube exponencialmente
        env = np.power(np.linspace(0, 1, len(t)), 2)
        return noise * env * 0.3
    @staticmethod
    def snare(sr=44100):
        """Sonido de caja (snare) basado en ruido blanco y un tono de 180Hz."""
        duration = 0.15
        t = np.linspace(0, duration, int(sr * duration))
        
        # Cuerpo del golpe (tonal)
        body = np.sin(2 * np.pi * 180 * np.exp(-20 * t) * t)
        # El "brillo" (ruido)
        snap = np.random.uniform(-1, 1, len(t)) * np.exp(-15 * t)
        
        return (body * 0.5 + snap * 0.5) * np.exp(-5 * t)