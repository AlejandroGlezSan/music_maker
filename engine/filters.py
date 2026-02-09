import numpy as np
from scipy import signal

class Filters:
    """
    ENGINE: FILTERS v12.0 - ANALOG MODELING & DSP
    --------------------------------------------
    Implementación de filtros de precisión con modelado de componentes.
    """

    @staticmethod
    def moog_ladder(data, cutoff, resonance, sr=44100):
        """
        Modelado digital del legendario filtro Moog de 24dB/octava.
        Añade saturación no lineal y una resonancia cálida.
        """
        # Normalización de frecuencia (0 a 1)
        f = (cutoff * 2.0) / sr
        k = 3.6 * f - 1.6 * f * f - 1.0
        p = (k + 1.0) * 0.5
        scale = np.exp((1.0 - p) * 1.386249)
        r = resonance * scale
        
        output = np.zeros(len(data))
        v = np.zeros(4) # 4 etapas del filtro
        
        for i in range(len(data)):
            # Saturación de entrada (tanh) para carácter analógico
            in_val = np.tanh(data[i] - 4.0 * r * v[3])
            
            # 4 etapas de integración
            v[0] = p * in_val + 0.3 * v[0]
            v[1] = p * v[0] + 0.3 * v[1]
            v[2] = p * v[1] + 0.3 * v[2]
            v[3] = p * v[2] + 0.3 * v[3]
            
            output[i] = v[3]
            
        return output

    @staticmethod
    def dynamic_envelope_filter(data, sensitivity=1.0, base_cutoff=500, sr=44100):
        """
        Filtro que reacciona a la amplitud (Auto-Wah).
        Ideal para darle vida a sintetizadores estáticos.
        """
        # Seguidor de envolvente (Rectificación + Suavizado)
        env = np.abs(data)
        # Filtro paso bajo para la envolvente (suavizado de 10Hz)
        b, a = signal.butter(1, 10 / (sr/2), 'low')
        smooth_env = signal.filtfilt(b, a, env)
        
        output = np.zeros_like(data)
        # Procesamos en bloques para eficiencia
        block_size = 128
        for i in range(0, len(data), block_size):
            end = min(i + block_size, len(data))
            # La frecuencia de corte sube con la intensidad del audio
            current_cutoff = base_cutoff + (smooth_env[i] * sensitivity * 5000)
            current_cutoff = min(current_cutoff, sr/2.1)
            
            # Aplicamos un filtro variable por bloque
            b, a = signal.butter(2, current_cutoff / (sr/2), 'low')
            output[i:end] = signal.lfilter(b, a, data[i:end])
            
        return output

    @staticmethod
    def dj_eq_3band(data, low_gain=1.0, mid_gain=1.0, high_gain=1.0, sr=44100):
        """
        Ecualizador de 3 bandas con crossovers fijos (400Hz y 4000Hz).
        Permite aislar o potenciar frecuencias como en una mesa de mezclas.
        """
        # Crossovers
        low_cut = 400
        high_cut = 4000
        
        # Filtros de separación
        b_low, a_low = signal.butter(2, low_cut / (sr/2), 'low')
        b_high, a_high = signal.butter(2, high_cut / (sr/2), 'high')
        
        low_band = signal.lfilter(b_low, a_low, data)
        high_band = signal.lfilter(b_high, a_high, data)
        mid_band = data - low_band - high_band
        
        return (low_band * low_gain) + (mid_band * mid_gain) + (high_band * high_gain)

    @staticmethod
    def resonant_high_pass(data, cutoff, resonance, sr=44100):
        """Filtro paso alto para limpiar sub-graves innecesarios."""
        # Usamos implementación de scipy para máxima estabilidad en HPF
        nyquist = sr / 2
        norm_cutoff = cutoff / nyquist
        # Q (Resonancia) convertida de 0-1 a escala Butterworth
        q_val = 0.707 + (resonance * 5.0)
        b, a = signal.butter(2, norm_cutoff, btype='highpass', analog=False)
        return signal.lfilter(b, a, data)

    # --- 5. UTILIDADES DE FORMANTES (VOZ ROBÓTICA) ---

    @staticmethod
    def formant_filter(data, vowel='a', sr=44100):
        """
        Refactorización del filtro de formantes.
        Utiliza picos de resonancia específicos para simular la garganta humana.
        """
        vowels = {
            'a': [(730, 1.0), (1090, 2.0), (2440, 0.5)],
            'e': [(360, 1.0), (2220, 2.0), (2960, 0.5)],
            'i': [(270, 1.0), (2290, 2.0), (3010, 0.5)],
            'o': [(400, 1.0), (840, 2.0), (2800, 0.5)],
            'u': [(300, 1.0), (870, 2.0), (2240, 0.5)]
        }
        
        v = vowel.lower()
        if v not in vowels: return data
        
        output = np.zeros_like(data)
        for freq, boost in vowels[v]:
            # Creamos un filtro de banda para cada formante
            b, a = signal.butter(2, [freq*0.8/(sr/2), freq*1.2/(sr/2)], btype='bandpass')
            output += signal.lfilter(b, a, data) * boost
            
        return output / 3.0