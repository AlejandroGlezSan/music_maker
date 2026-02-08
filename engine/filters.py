import numpy as np

class Filters:
    @staticmethod
    def resonant_low_pass(data, cutoff, resonance, sr=44100):
        """
        Filtro IIR de paso bajo con resonancia (simulación básica).
        cutoff: Frecuencia de corte en Hz.
        resonance: Factor Q (0.0 a 0.99). Cuanto más alto, más 'chillón'.
        """
        # Coeficientes simplificados para un filtro de estado variable
        c = 2.0 * np.sin(np.pi * cutoff / sr)
        q = resonance
        
        low = np.zeros(len(data))
        band = np.zeros(len(data))
        high = np.zeros(len(data))
        
        l, b = 0.0, 0.0
        
        # Procesamiento muestra a muestra (Lento en Python puro, pero efectivo)
        # Nota: En producción real esto se hace en C++, pero sirve para el demo.
        for i in range(len(data)):
            l = l + c * b
            h = data[i] - l - q * b
            b = b + c * h
            low[i] = l
            
        return low