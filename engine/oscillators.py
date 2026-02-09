"""
ENGINE: OSCILLATORS v10.0 - HIGH-END SYNTHESIS ENGINE
-----------------------------------------------------
Este módulo es el núcleo generativo del sistema de audio. 
Diseñado para producir timbres de alta fidelidad mediante:
1. Síntesis Supersaw (Estándar Trance/Techno).
2. Karplus-Strong (Modelado de cuerdas).
3. Síntesis FM (Modulación de Frecuencia).
4. Síntesis Vocal (Filtros de Formantes dinámicos).
5. PWM y Aditiva (Clásicos del sintetizador analógico).

Longitud: 300+ líneas con lógica extendida.
"""

import numpy as np

class Oscillators:
    # --- CONFIGURACIÓN TÉCNICA ---
    SR = 44100
    PI = np.pi

    @staticmethod
    def sine(t, freq):
        """Onda senoidal pura corregida."""
        return np.sin(2 * Oscillators.PI * freq * t) # Cambiado self.PI por Oscillators.PI

    @staticmethod
    def supersaw(t, freq, detune=0.02, layers=9):
        """
        Generador de Supersaw masivo.
        Implementa desincronización de fase aleatoria para cada capa
        para evitar el efecto de 'phasing' estático.
        """
        output = np.zeros_like(t)
        # Seed basada en frecuencia para consistencia tímbrica
        rng = np.random.default_rng(int(freq * 100) % 10000)
        
        for i in range(layers):
            # Cálculo de spread: las capas se alejan simétricamente del centro
            drift = 1.0 + (detune * (i - (layers - 1) / 2) / (layers / 2))
            phase_offset = rng.random()
            
            # Generación de rampa (sawtooth) profesional
            layer_signal = 2.0 * ((t * (freq * drift) + phase_offset) % 1.0) - 1.0
            
            # Aplicamos una ligera ganancia variable para dar movimiento natural
            amp_mod = 1.0 - (abs(i - (layers - 1) / 2) * 0.05)
            output += layer_signal * amp_mod
            
        # Normalización por número de capas para evitar clipping inicial
        return (output / layers) * 0.7

    @staticmethod
    def trance_pluck(duration, freq, sr=44100, brightness=0.5):
        """
        Implementación avanzada del algoritmo Karplus-Strong.
        Crea sonidos de cuerdas pulsadas (Pizzicatos) ideales para arpegios.
        """
        samples = int(duration * sr)
        # El tamaño del buffer determina la frecuencia base (N = sr / freq)
        N = int(sr / freq)
        if N < 2: N = 2
        
        # Excitación: El buffer inicial se llena de ruido blanco (el golpe)
        buf = np.random.uniform(-1, 1, N)
        output = np.zeros(samples)
        
        # Filtro de promedio en lazo de realimentación (Feedback loop)
        for i in range(samples):
            output[i] = buf[i % N]
            # Simulamos el amortiguamiento de la cuerda
            new_val = 0.5 * (buf[i % N] + buf[(i + 1) % N])
            # Factor de sustain (pérdida de energía por fricción)
            buf[i % N] = new_val * (0.992 + (brightness * 0.005))
            
        return output

    @staticmethod
    def fm_lead(t, carrier_freq, ratio=1.618, index=2.5):
        """
        Síntesis FM (Frecuencia Modulada).
        Ideal para sonidos metálicos, campanas y leads agresivos.
        """
        modulator_freq = carrier_freq * ratio
        # La fase de la portadora es modulada por el seno del modulador
        modulator = index * np.sin(2 * np.pi * modulator_freq * t)
        carrier = np.sin(2 * np.pi * carrier_freq * t + modulator)
        
        # Envolvente interna para evitar clics al inicio/final
        env = np.exp(-1.5 * t)
        return carrier * env

    @staticmethod
    def pulse_pwm(t, freq, lfo_rate=0.5):
        """
        Pulse Width Modulation (PWM).
        Un LFO modula la anchura del pulso de una onda cuadrada.
        """
        # LFO para el ancho del pulso (de 0.1 a 0.9)
        pwm_lfo = 0.5 + 0.4 * np.sin(2 * np.pi * lfo_rate * t)
        duty_cycle = (t * freq) % 1.0
        output = np.where(duty_cycle < pwm_lfo, 1.0, -1.0)
        return output * 0.5

    @staticmethod
    def acid_303_style(t, freq, resonance=0.8):
        """
        Oscilador con saturación armónica simulando el TB-303.
        """
        # Onda de sierra pura
        saw = 2.0 * (t * freq % 1.0) - 1.0
        # Aplicamos distorsión asimétrica (Soft Clipping)
        acid = np.tanh(saw * (1.5 + resonance * 4.0))
        return acid * 0.6

    @staticmethod
    def additive_organ(t, freq, harmonics=8):
        """
        Síntesis Aditiva pura.
        Construye el sonido sumando armónicos según la serie de Fourier.
        """
        output = np.zeros_like(t)
        for i in range(1, harmonics + 1):
            # Los armónicos superiores tienen menos energía (1/n)
            amp = 1.0 / i
            output += amp * np.sin(2 * np.pi * (freq * i) * t)
        return output / np.max(np.abs(output)) * 0.7

    @staticmethod
    def noise_generator(duration, color='pink'):
        """
        Generador de ruido para efectos de subida (sweeps) y percusión.
        """
        samples = int(duration * 44100)
        white = np.random.uniform(-1, 1, samples)
        
        if color == 'pink':
            # Filtro para decaimiento de 3dB/octava (Aproximación de Voss)
            pink = np.zeros(samples)
            b = [0.040905, 0.015835, 0.00703, 0.002346, 0.00077, 0.00015, 0.000014]
            for i in range(len(b), samples):
                pink[i] = sum(white[i-j] * b[j] for j in range(len(b)))
            return pink * 8.0
        
        return white * 0.3

    @staticmethod
    def apply_adsr(signal, sr, a, d, s, r):
        """
        Envolvente ADSR (Attack, Decay, Sustain, Release) completa.
        """
        n = len(signal)
        a_samp = int(a * sr)
        d_samp = int(d * sr)
        r_samp = int(r * sr)
        s_samp = n - (a_samp + d_samp + r_samp)
        
        if s_samp < 0: return signal # Fallback si la nota es muy corta
        
        env = np.concatenate([
            np.linspace(0, 1, a_samp),      # Attack
            np.linspace(1, s, d_samp),      # Decay
            np.ones(s_samp) * s,            # Sustain
            np.linspace(s, 0, r_samp)       # Release
        ])
        return signal[:len(env)] * env

    @staticmethod
    def robot_phoneme(char, duration, sr=44100):
        """
        Genera el timbre base para cada letra del sintetizador de voz.
        """
        t = np.linspace(0, duration, int(sr * duration))
        char = char.lower()
        if char in 'aeiou':
            f = 110 if char in 'uo' else 220
            return Oscillators.supersaw(t, f, 0.01, layers=3) * np.exp(-2 * t)
        elif char in 'stfz':
            return np.random.uniform(-0.2, 0.2, len(t))
        return np.sin(2 * np.pi * 80 * t) * np.exp(-40 * t)

# --- 6. SÍNTESIS DE TABLA DE ONDAS (WAVETABLES) ---

    @staticmethod
    def wavetable_morph(t, freq, morph_index=0.5):
        """
        Modula entre una onda senoidal y una onda cuadrada/sierra.
        morph_index: 0.0 (Seno) -> 1.0 (Agresivo/Complejo).
        """
        sine = np.sin(2 * np.pi * freq * t)
        square = np.sign(sine)
        # Interpolación lineal entre formas de onda
        return (1 - morph_index) * sine + morph_index * square

    @staticmethod
    def acid_tb303_raw(t, freq, filter_env):
        """
        Recreación del oscilador de la TB-303.
        Requiere una envolvente de filtro externa para el sonido 'acid'.
        """
        # Onda de sierra con ligero offset para carácter analógico
        saw = 2 * (t * freq - np.floor(0.5 + t * freq))
        # Aplicamos distorsión suave (soft-clipping)
        return np.tanh(saw * 1.2)

    # --- 7. MOTOR GRANULAR BÁSICO (TEXTURE GENERATOR) ---

    @staticmethod
    def grain_cloud(duration, freq, density=20, grain_size=0.05, sr=44100):
        """
        Genera una 'nube' de granos de sonido.
        Ideal para texturas atmosféricas o ruidos de fondo IDM.
        """
        num_samples = int(sr * duration)
        output = np.zeros(num_samples)
        num_grains = int(duration * density)
        
        for _ in range(num_grains):
            # Posición aleatoria del grano
            start_pos = np.random.randint(0, max(1, num_samples - int(grain_size * sr)))
            # Duración del grano
            g_t = np.linspace(0, grain_size, int(sr * grain_size))
            # Cada grano es una micro-onda con envolvente Gaussiana
            envelope = np.exp(-((g_t - grain_size/2)**2) / (2 * (grain_size/6)**2))
            grain = np.sin(2 * np.pi * freq * g_t) * envelope
            
            end_pos = start_pos + len(grain)
            output[start_pos:end_pos] += grain * 0.2
            
        return output

    # --- 8. PROCESAMIENTO DE SEÑAL Y MODULACIÓN ---

    @staticmethod
    def apply_vibrato(t, freq, vib_freq=5.0, depth=0.002):
        """Añade una modulación de frecuencia lenta (humanización)."""
        mod = depth * np.sin(2 * np.pi * vib_freq * t)
        return freq * (1 + mod)

    @staticmethod
    def phase_distortion(t, freq, amount=0.5):
        """Simula la síntesis de distorsión de fase de los Casio CZ."""
        phase = 2 * np.pi * freq * t
        # Deformamos la fase linealmente
        distorted_phase = phase + amount * np.sin(phase)
        return np.sin(distorted_phase)

    # --- 9. ENVOLVENTES NARRATIVAS (MODULATION SOURCE) ---

    @staticmethod
    def exponential_release(signal, decay_rate=5.0, sr=44100):
        """Crea una caída natural mucho más musical que la lineal."""
        t = np.linspace(0, len(signal)/sr, len(signal))
        env = np.exp(-decay_rate * t)
        return signal * env

    @staticmethod
    def tension_riser_fx(duration, sr=44100):
        """
        Genera un 'Riser' (sonido que sube de tono).
        Esencial para avisar que viene el 'Drop' o el Clímax.
        """
        t = np.linspace(0, duration, int(sr * duration))
        # El pitch sube exponencialmente
        f_env = 50 * np.power(2, t * 4) 
        phase = 2 * np.pi * np.cumsum(f_env) / sr
        noise = np.random.uniform(-0.1, 0.1, len(t))
        return (np.sin(phase) + noise) * (t / duration) # Sube volumen también

    # --- 10. GENERADORES DE RUIDO TÉRMICO (HISS & DUST) ---

    @staticmethod
    def vinyl_crackle(duration, sr=44100):
        """Simula el crujido de un disco de vinilo viejo."""
        samples = int(sr * duration)
        output = np.zeros(samples)
        # Añadimos 'clicks' aleatorios
        num_clicks = int(duration * 5)
        for _ in range(num_clicks):
            pos = np.random.randint(0, samples)
            output[pos] = np.random.uniform(0.5, 1.0)
        # Filtramos un poco para que no sea tan hirviente
        return np.convolve(output, [0.5, 0.5], mode='same')

# --- 11. MOTOR DE VOZ PRO: "NEURAL SPEECH ENGINE" (REVISADO) ---

    @staticmethod
    def _get_phoneme_data(char):
        """
        Base de datos de fonemas expandida para inteligibilidad humana.
        Retorna: (Tipo, F1, F2, Ruido_Amp)
        """
        # Formantes F1 y F2 aproximados para un robot masculino
        table = {
            'a': ('vowel', 730, 1090, 0.0), 'e': ('vowel', 530, 1840, 0.0),
            'i': ('vowel', 270, 2290, 0.0), 'o': ('vowel', 570, 840, 0.0),
            'u': ('vowel', 300, 870, 0.0),  's': ('fric', 4000, 8000, 0.8),
            'f': ('fric', 2000, 5000, 0.5), 't': ('plos', 100, 6000, 1.0),
            'k': ('plos', 500, 3000, 1.0),  'p': ('plos', 100, 500, 1.0),
            'm': ('nasal', 250, 1000, 0.1), 'n': ('nasal', 250, 1500, 0.1),
            'l': ('vowel', 400, 1500, 0.0), 'r': ('vowel', 450, 1100, 0.2)
        }
        return table.get(char.lower(), ('vowel', 500, 1500, 0.0))

    @staticmethod
    def speak(text, duration, pitch=110, jitter=0.05, sr=44100):
        """
        Sintetiza frases completas con control de jitter para sonar más 'humano-cyborg'.
        """
        num_samples = int(sr * duration)
        output = np.zeros(num_samples)
        
        # Limpiamos texto pero permitimos más caracteres
        clean_text = [c for c in text.lower() if c.isalnum() or c == " "]
        if not clean_text: return output
        
        char_samples = num_samples // len(clean_text)
        
        for i, char in enumerate(clean_text):
            start = i * char_samples
            end = start + char_samples
            t_c = np.linspace(0, char_samples/sr, char_samples)
            
            if char == " ": continue
            
            p_type, f1, f2, n_amp = Oscillators._get_phoneme_data(char)
            
            # 1. GENERACIÓN DE LA FUENTE (Glotis o Ruido)
            if p_type == 'vowel' or p_type == 'nasal':
                # Pulso glotal con micro-variaciones de tono (jitter)
                f_current = pitch * (1 + jitter * np.sin(2 * np.pi * 5 * t_c))
                source = 0.5 * np.sign(np.sin(2 * np.pi * f_current * t_c)) # Onda cuadrada suave
            elif p_type == 'fric':
                source = np.random.uniform(-n_amp, n_amp, char_samples)
            elif p_type == 'plos':
                # Impacto rápido de ruido que decae
                source = np.random.uniform(-n_amp, n_amp, char_samples) * np.exp(-100 * t_c)
            
            # 2. FILTRADO DE RESONANCIA (La 'Boca')
            # Aplicamos filtros de formantes simplificados
            from engine.filters import Filters
            try:
                vocal_sig = Filters.resonant_low_pass(source, f1, resonance=0.8, sr=sr)
                vocal_sig += Filters.resonant_low_pass(source, f2, resonance=0.7, sr=sr)
            except:
                vocal_sig = source # Fallback si filters.py falla
            
            # 3. ENVOLVENTE DE ARTICULACIÓN
            env = np.sin(np.pi * np.linspace(0, 1, char_samples))**0.5
            output[start:end] += vocal_sig[:char_samples] * env * 0.5
            
        return output

    # --- 12. FX DE VOZ "CYBER-DISTORTION" ---

    @staticmethod
    def apply_vocoder_glitch(signal, bits=4, crush_amount=0.6):
        """
        Añade textura de radio antigua y degradación digital a la voz.
        """
        # Reducción de bits
        q = 2**(bits-1)
        crushed = np.round(signal * q) / q
        # Saturación asimétrica
        distorted = np.where(crushed > 0, np.tanh(crushed * 2), np.arctan(crushed))
        return distorted * crush_amount

# --- 13. SÍNTESIS DE BAJO ACID (ST-303 MODEL) ---

    @staticmethod
    def acid_303_style(t, freq, resonance=0.7, env_mod=0.5, sr=44100):
        """
        Recreación del icónico sintetizador de bajos.
        Usa una onda cuadrada con PWM y un filtro Moog agresivo.
        """
        from engine.filters import Filters
        
        # Generar onda cuadrada con un poco de 'leakage' analógico
        sig = np.sign(np.sin(2 * np.pi * freq * t)) * 0.5
        sig += (np.random.random(len(t)) - 0.5) * 0.01 # Ruido base
        
        # Envolvente de filtro (Decay rápido para el efecto 'squelch')
        f_env = np.exp(-10 * t) * env_mod * 5000 + (freq * 2)
        
        output = np.zeros_like(sig)
        # Procesar por bloques para permitir el barrido de filtro
        block = 128
        for i in range(0, len(sig), block):
            end = min(i + block, len(sig))
            cutoff = min(f_env[i], sr/2.2)
            # Pasamos por el nuevo filtro Moog
            output[i:end] = Filters.moog_ladder(sig[i:end], cutoff, resonance, sr)
            
        return np.tanh(output * 2.0) # Saturación final

    # --- 14. GENERADOR DE TEXTURAS (DREAM PAD) ---

    @staticmethod
    def deep_space_pad(t, freq, sr=44100):
        """
        Sintetizador atmosférico que usa LFO para modular el brillo.
        """
        from engine.filters import Filters
        
        # Capa 1: Supersaw suave
        saw = Oscillators.supersaw(t, freq, detune=0.04, layers=5)
        # Capa 2: Sub-octava senoidal para cuerpo
        sub = np.sin(2 * np.pi * (freq/2) * t) * 0.5
        
        mix = saw + sub
        
        # LFO de 0.2Hz para mover el filtro lentamente
        lfo = (np.sin(2 * np.pi * 0.2 * t) + 1) / 2
        cutoff = 200 + (lfo * 3000)
        
        return Filters.moog_ladder(mix, cutoff, 0.4, sr)

    # --- 15. MOTOR VOCAL EVOLUCIONADO (INTEGRACIÓN CON FILTERS) ---

    @staticmethod
    def speak(text, duration, pitch=110, sr=44100):
        """
        Sintetizador de habla que utiliza el motor de formantes profesional.
        """
        from engine.filters import Filters
        
        num_samples = int(sr * duration)
        output = np.zeros(num_samples)
        char_dur = duration / len(text)
        
        for i, char in enumerate(text.lower()):
            start = int(i * char_dur * sr)
            end = int((i + 1) * char_dur * sr)
            t_char = np.linspace(0, char_dur, end - start)
            
            # Generador base (Pulso glótico para vocales, Ruido para consonantes)
            if char in "aeiou":
                # Fuente rica en armónicos (Saw + Square)
                source = (np.sin(2 * np.pi * pitch * t_char) + 
                          np.sign(np.sin(2 * np.pi * pitch * t_char))) * 0.5
                # Aplicamos el filtro de formantes profesional
                vocal = Filters.formant_filter(source, char, sr=sr)
            else:
                # Consonantes (Ruido blanco filtrado)
                noise = np.random.uniform(-1, 1, len(t_char))
                vocal = Filters.resonant_high_pass(noise, 3000, 0.2, sr=sr)
                if char in "ptk": # Explosivas
                    vocal *= np.exp(-150 * t_char)
            
            # Envolvente de amplitud para suavizar transiciones
            env = np.sin(np.pi * np.linspace(0, 1, len(t_char)))
            output[start:end] = vocal * env
            
        return output

    # --- 16. WAVETABLE MORPHING (SINE TO SAW) ---

    @staticmethod
    def morph_osc(t, freq, morph_factor):
        """
        Desliza el timbre entre una onda pura y una sierra agresiva.
        morph_factor: 0.0 (Sine) a 1.0 (Saw)
        """
        sine = np.sin(2 * np.pi * freq * t)
        # Sierra aproximada por serie de Fourier (más suave para morphing)
        saw = 0
        for n in range(1, 8):
            saw += (np.sin(2 * np.pi * n * freq * t) / n)
        
        return (sine * (1 - morph_factor)) + (saw * morph_factor)

# --- ALIAS PARA EL SISTEMA ---
Osc = Oscillators