import numpy as np

class Effects:
    """
    Módulo de post-procesamiento. 
    Añade profundidad, espacio y control de dinámica.
    """

    @staticmethod
    def reverb_simple(data, room_size=0.5, damp=0.5, sr=44100):
        """Reverberación basada en múltiples líneas de retraso (estilo Schroeder)."""
        # Crear 4 retrasos diferentes para simular reflexiones
        delays = [int(0.029 * sr), int(0.037 * sr), int(0.044 * sr), int(0.050 * sr)]
        output = np.copy(data)
        for d in delays:
            reverb_layer = np.zeros_like(data)
            if d < len(data):
                reverb_layer[d:] = data[:-d] * room_size
            output += reverb_layer
        return output * 0.6

    @staticmethod
    def bitcrusher(data, bits=8):
        """Distorsión digital reduciendo la resolución de bits."""
        q = 2**(bits-1)
        return np.round(data * q) / q

    @staticmethod
    def sidechain_compressor(data, trigger, threshold=0.1, ratio=4):
        """Reduce el volumen de la data cuando el trigger (Kick) está alto."""
        envelope = np.abs(trigger)
        gain = np.ones_like(data)
        for i in range(len(envelope)):
            if envelope[i] > threshold:
                gain[i] = 1.0 / ratio
        # Suavizar la transición de la ganancia
        gain = np.convolve(gain, np.ones(500)/500, mode='same')
        return data * gain

    @staticmethod
    def ping_pong_delay(data, delay_ms=300, feedback=0.4, sr=44100):
        """Eco que rebota (simulado en mono para este motor)."""
        d_samples = int((delay_ms / 1000) * sr)
        output = np.copy(data)
        for i in range(d_samples, len(data), d_samples):
            segment = data[i-d_samples:i] * feedback
            if i + len(segment) < len(data):
                output[i:i+len(segment)] += segment
            feedback *= 0.8
        return output

    @staticmethod
    def master_limiter(data, ceiling=0.95):
        """Evita que el audio distorsione al superar el 0dB."""
        max_val = np.max(np.abs(data))
        if max_val > ceiling:
            return (data / max_val) * ceiling
        return data

# --- 6. PROCESAMIENTO ESPACIAL Y CINEMATOGRÁFICO ---

    @staticmethod
    def stereo_widener(data, width=1.5):
        """
        Crea amplitud estéreo (simulado en mono mediante delay de Haas).
        Ideal para que la voz robótica se sienta 'grande'.
        """
        delay_samples = int(0.02 * 44100) # 20ms de desfase
        if len(data) < delay_samples: return data
        
        # Generamos una señal con retardo para ensanchar el centro
        delayed = np.zeros_like(data)
        delayed[delay_samples:] = data[:-delay_samples]
        
        return (data + delayed * (width - 1.0)) / width

    @staticmethod
    def cinematic_reverb(data, decay=0.8, density=0.7, sr=44100):
        """
        Reverb de alta densidad para atmósferas épicas e industriales.
        Usa múltiples líneas de retraso para evitar el sonido metálico.
        """
        output = np.copy(data)
        # Tiempos de retraso basados en números primos
        delays = [int(ms * sr / 1000) for ms in [149, 107, 73, 41]]
        
        for d in delays:
            layer = np.zeros_like(data)
            for i in range(1, 4): # Generamos 3 reflexiones por capa
                pos = d * i
                if pos < len(data):
                    layer[pos:] += data[:-pos] * (decay ** i)
            output += layer * (density / 4.0)
            
        return Effects.master_limiter(output * 0.5)

    # --- 7. MODULACIÓN DE VOCODER Y TEXTURA ---

    @staticmethod
    def vocoder_shaper(carrier, modulator, sensitivity=0.8):
        """
        EL MOTOR DEL ROBOT: Aplica la envolvente de la voz al sintetizador.
        Carrier: El sonido del synth. Modulator: La voz humana/robot.
        """
        # Extraer envolvente del modulador (la voz)
        envelope = np.abs(modulator)
        # Suavizado para evitar 'clicks'
        window = 800
        kernel = np.ones(window) / window
        smooth_env = np.convolve(envelope, kernel, mode='same')
        
        # Normalización y aplicación al synth
        norm_env = (smooth_env / (np.max(smooth_env) + 1e-8)) * sensitivity
        return carrier * norm_env

    @staticmethod
    def chorus(data, rate=1.2, depth=0.002, sr=44100):
        """Añade grosor duplicando la señal con micro-desafinaciones."""
        t = np.linspace(0, len(data)/sr, len(data))
        lfo = depth * np.sin(2 * np.pi * rate * t)
        indices = np.arange(len(data))
        mod_idx = np.clip(indices - (lfo * sr).astype(int), 0, len(data)-1)
        return (data + data[mod_idx]) * 0.5

    # --- 8. DINÁMICA DE MASTERIZACIÓN Y CALOR ---

    @staticmethod
    def tube_warmth(data, drive=2.5, warmth=0.4):
        """Simula la saturación de una válvula para dar 'alma' al audio."""
        # Recorte asimétrico para armónicos musicales
        driven = np.where(data > 0, 
                         np.tanh(data * drive), 
                         np.tanh(data * drive * (1-warmth)))
        return driven

    @staticmethod
    def adaptive_compressor(data, threshold=0.15, ratio=4.0, attack=0.01, sr=44100):
        """
        Controla los picos de volumen para que la mezcla no sature 
        y la voz siempre sea inteligible.
        """
        output = np.zeros_like(data)
        gain = 1.0
        alpha = np.exp(-1.0 / (attack * sr))
        
        for i in range(len(data)):
            env = np.abs(data[i])
            if env > threshold:
                target = threshold + (env - threshold) / ratio
                target /= env
            else:
                target = 1.0
            
            gain = alpha * gain + (1 - alpha) * target
            output[i] = data[i] * gain
            
        return output

    @staticmethod
    def tilt_eq(data, brightness=0.2):
        """Ajusta el balance tonal: añade brillo o quita oscuridad."""
        # Refuerzo de agudos mediante la diferencia de señal
        highs = np.diff(data, prepend=0)
        return data + highs * brightness

# Alias
Ef = Effects