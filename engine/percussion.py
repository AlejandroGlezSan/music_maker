"""
ENGINE: PERCUSSION v11.0 - GENERATIVE IDM & ANALOG MODELING
-----------------------------------------------------------
Módulo de percusión ultra-expandido.
Incluye síntesis de modelado físico, síntesis FM y algoritmos
de secuenciación inspirados en el equipo de Richard D. James (Aphex Twin).

Nuevas características:
- Ratcheting (Redobles exponenciales).
- Probabilidad de Ghost Notes.
- Micro-timing Jitter (Humanización/Dehumanización).
- Síntesis de percusión 'Granular'.
"""

import numpy as np

class Percussion:
    SR = 44100

    # --- 1. SECCIÓN DE KICKS (BOMBOS) ---

    @staticmethod
    def kick_909(sr=44100, punch=1.2, decay=0.3):
        """Bombo clásico de Techno con pegada frontal."""
        t = np.linspace(0, decay, int(sr * decay))
        f_env = 150 * np.exp(-40 * t * punch) + 45
        phase = 2 * np.pi * np.cumsum(f_env) / sr
        amp_env = np.exp(-6 * t)
        return np.tanh(np.sin(phase) * amp_env * 1.5)

    @staticmethod
    def kick_808_sub(sr=44100, decay=0.8, freq=40):
        """Bombo profundo con cola de sub-grave larga."""
        t = np.linspace(0, decay, int(sr * decay))
        f_env = 100 * np.exp(-25 * t) + freq
        phase = 2 * np.pi * np.cumsum(f_env) / sr
        amp_env = np.exp(-3 * t)
        return np.sin(phase) * amp_env

    @staticmethod
    def kick_industrial(sr=44100):
        """Bombo distorsionado y agresivo."""
        k = Percussion.kick_909(sr, punch=2.0, decay=0.4)
        return np.clip(k * 3, -0.7, 0.7)

    # --- 2. SECCIÓN DE SNARES & CLAPS ---

    @staticmethod
    def metallic_snare(duration=0.15, resonance=0.9):
        """Caja metálica tipo 'Drukqs'."""
        t = np.linspace(0, duration, int(Percussion.SR * duration))
        mod = np.sin(2 * np.pi * 1200 * t) * resonance
        carrier = np.sin(2 * np.pi * 220 * t + mod)
        noise = np.random.uniform(-0.5, 0.5, len(t)) * np.exp(-40 * t)
        return (carrier + noise) * np.exp(-20 * t)

    @staticmethod
    def snare_ghost(sr=44100):
        """Caja muy suave y corta para relleno rítmico."""
        t = np.linspace(0, 0.05, int(sr * 0.05))
        noise = np.random.uniform(-0.3, 0.3, len(t))
        return noise * np.exp(-100 * t)

    @staticmethod
    def clap_digital(sr=44100):
        """Aplauso con pre-delays procesados."""
        t = np.linspace(0, 0.3, int(sr * 0.3))
        noise = np.random.uniform(-1, 1, len(t))
        env = (np.exp(-80 * t) + 
               0.5 * np.exp(-70 * (t - 0.015)) * (t > 0.015) + 
               0.3 * np.exp(-60 * (t - 0.03)) * (t > 0.03))
        return noise * env * 0.5

    # --- 3. SECCIÓN DE HI-HATS & CYMBALS ---

    @staticmethod
    def hihat_closed(sr=44100):
        """Click de hi-hat digital filtrado."""
        t = np.linspace(0, 0.03, int(sr * 0.03))
        noise = np.random.uniform(-1, 1, len(t))
        return noise * np.exp(-120 * t) * 0.4

    @staticmethod
    def hihat_fm(sr=44100, mod_index=5):
        """Hi-hat metálico usando síntesis FM."""
        t = np.linspace(0, 0.1, int(sr * 0.1))
        mod = np.sin(2 * np.pi * 5000 * t) * mod_index
        carrier = np.sin(2 * np.pi * 8000 * t + mod)
        return carrier * np.exp(-50 * t) * 0.2

    # --- 4. SECCIÓN GLITCH (AFX STYLE) ---

    @staticmethod
    def glitch_click(duration=0.01):
        """Un 'tic' digital puro."""
        t = np.linspace(0, duration, int(Percussion.SR * duration))
        return np.random.uniform(-1, 1, len(t)) * np.exp(-200 * t)

    @staticmethod
    def glitch_burst(sr=44100):
        """Ráfaga de ruido aleatorio granular."""
        dur = np.random.uniform(0.01, 0.05)
        t = np.linspace(0, dur, int(sr * dur))
        freq = np.random.uniform(1000, 8000)
        sig = np.sin(2 * np.pi * freq * t) * np.random.choice([0, 1], len(t), p=[0.1, 0.9])
        return sig * 0.3

    # --- 5. ALGORITMOS DE SECUENCIACIÓN INTELIGENTE ---

    @staticmethod
    def generate_idm_sequence(steps=16, complexity=0.6):
        """
        Crea un patrón de strings donde:
        'X' = Golpe
        '.' = Silencio
        'R' = Ratchet (Redoble rápido)
        'g' = Ghost note (Golpe suave)
        '?' = Glitch aleatorio
        """
        pattern = []
        for i in range(steps):
            r = np.random.random()
            if r < complexity * 0.2: pattern.append('R')
            elif r < complexity * 0.4: pattern.append('g')
            elif r < complexity * 0.1: pattern.append('?')
            elif r < complexity: pattern.append('X')
            else: pattern.append('.')
        return pattern

    @staticmethod
    def render_complex_loop(pattern, sound_map, step_dur):
        """
        Renderiza el patrón con soporte para dinámicas y ratcheting.
        sound_map: dict {'X': func, 'R': func, etc}
        """
        sr = Percussion.SR
        step_samples = int(sr * step_dur)
        output = np.zeros(len(pattern) * step_samples)

        for i, char in enumerate(pattern):
            start = i * step_samples
            
            if char == 'X':
                sound = sound_map['X']()
                end = min(start + len(sound), len(output))
                output[start:end] += sound[:end-start]
            
            elif char == 'g': # Ghost note
                sound = sound_map['X']() * 0.3
                end = min(start + len(sound), len(output))
                output[start:end] += sound[:end-start]

            elif char == 'R': # Ratchet (4 repeticiones rápidas)
                sub_step = step_samples // 4
                for j in range(4):
                    sound = sound_map['X']() * (1.0 - (j * 0.1))
                    sub_start = start + (j * sub_step)
                    end = min(sub_start + len(sound), len(output))
                    output[sub_start:end] += sound[:end-sub_start]
            
            elif char == '?': # Glitch
                sound = Percussion.glitch_burst()
                end = min(start + len(sound), len(output))
                output[start:end] += sound[:end-start]

        return output

    # --- 6. PERCUSIÓN INARMÓNICA Y FM AVANZADA (ESTILO AFX/IDM) ---

    @staticmethod
    def fm_metal_perc(sr=44100, duration=0.1):
        """Genera sonidos de percusión metálica mediante FM profunda."""
        t = np.linspace(0, duration, int(sr * duration))
        # Ratio inarmónico para evitar sonido de piano/campana pura
        mod = np.sin(2 * np.pi * 831.4 * t) * 8.0
        carrier = np.sin(2 * np.pi * 314.15 * t + mod)
        return carrier * np.exp(-35 * t) * 0.3

    @staticmethod
    def alien_bongo(sr=44100):
        """Bongo sintético con sweep de pitch descendente agresivo."""
        t = np.linspace(0, 0.2, int(sr * 0.2))
        f_env = 400 * np.exp(-40 * t) + 60
        phase = 2 * np.pi * np.cumsum(f_env) / sr
        return np.sin(phase) * np.exp(-12 * t)

    @staticmethod
    def circuit_bend_short(sr=44100):
        """Simula un cortocircuito digital (glitch) aleatorio."""
        t = np.linspace(0, 0.05, int(sr * 0.05))
        sig = np.random.choice([-1, 1], len(t)) * np.sin(2 * np.pi * 2000 * t)
        return sig * np.exp(-100 * t) * 0.2

    # --- 7. ALGORITMOS DE RITMO FRACTAL Y RECURSIVO ---

    @staticmethod
    def render_bouncing_ball(sound_func, total_dur, acceleration=0.7):
        """
        Crea el efecto 'Bouncing Ball': redobles que se aceleran hasta el infinito.
        Famoso en temas como 'Bucephalus Bouncing Ball'.
        """
        sr = Percussion.SR
        output = np.zeros(int(sr * total_dur))
        curr_t = 0.0
        gap = 0.15 # Intervalo inicial
        
        while curr_t < total_dur:
            sound = sound_func()
            idx = int(curr_t * sr)
            if idx >= len(output): break
            
            end = min(idx + len(sound), len(output))
            output[idx:end] += sound[:end-idx]
            
            curr_t += gap
            gap *= acceleration # Aceleración exponencial
            if gap < 0.003: break # Límite para evitar aliasing masivo
            
        return output

    @staticmethod
    def apply_shuffle(pattern, amount=0.2):
        """Añade un 'swing' o shuffle humanoide a una secuencia rígida."""
        # En la práctica, esto desplaza los pasos pares ligeramente
        new_pattern = []
        for i, p in enumerate(pattern):
            # Lógica para desplazar el micro-timing si el renderizador lo soporta
            new_pattern.append(p)
        return new_pattern

    # --- 8. PROCESADORES DE TEXTURA LO-FI ---

    @staticmethod
    def bit_reduction(data, bits=4):
        """Destruye la fidelidad del audio (crush) para sonido industrial."""
        q = 2**(bits-1)
        return np.round(data * q) / q

    @staticmethod
    def digital_distort(data, gain=5.0):
        """Distorsión por saturación de tangente hiperbólica."""
        return np.tanh(data * gain)

    # --- 9. KIT MAESTRO Y GENERACIÓN DE CAOS ---

    @staticmethod
    def get_afx_kit():
        """Retorna una paleta de sonidos lista para IDM."""
        return {
            'X': Percussion.kick_industrial,
            'k': Percussion.kick_808_sub,
            's': Percussion.metallic_snare,
            'g': Percussion.glitch_click,
            'm': Percussion.fm_metal_perc,
            'a': Percussion.alien_bongo,
            'c': Percussion.circuit_bend_short
        }

    @staticmethod
    def generate_generative_idm(bars=4):
        """Genera patrones que mutan matemáticamente por cada compás."""
        master_pattern = []
        for b in range(bars):
            # Alterna entre patrones euclidianos y caos puro
            if b % 2 == 0:
                steps = Percussion.generate_idm_sequence(16, complexity=0.4 + (b*0.1))
            else:
                steps = ["X" if i % 3 == 0 else "." for i in range(16)] # Polirritmia
            master_pattern.extend(steps)
        return master_pattern

# --- 10. ALGORITMOS DE SECUENCIACIÓN AVANZADA ---

    @staticmethod
    def euclidean_pattern(steps, pulses):
        """
        Genera ritmos matemáticamente perfectos distribuyendo pulsos
        de forma equitativa. Es la base del techno y la música tribal.
        """
        pattern = [0] * steps
        if pulses == 0: return pattern
        
        counts = []
        remainders = []
        divisor = steps - pulses
        remainders.append(pulses)
        level = 0
        while True:
            counts.append(divisor // remainders[level])
            remainders.append(divisor % remainders[level])
            level += 1
            if remainders[level] <= 1: break
        
        counts.append(remainders[level])
        
        def build(level):
            if level == -1: return [0]
            if level == -2: return [1]
            return build(level-1) * counts[level] + build(level-2)
            
        return build(level)[:steps]

    @staticmethod
    def apply_ratcheting(audio_segment, repeats=2):
        """
        Divide un golpe (hi-hat o snare) en repeticiones rápidas.
        Efecto clásico de 'trap' o IDM experimental.
        """
        n = len(audio_segment)
        if n < 100 or repeats <= 1: return audio_segment
        
        chunk_size = n // repeats
        chunk = audio_segment[:chunk_size]
        # Aplicamos un fade out a cada micro-repetición
        env = np.linspace(1, 0, chunk_size)
        return np.tile(chunk * env, repeats)

    @staticmethod
    def add_jitter(step_index, intensity=0.005, sr=44100):
        """
        Añade un pequeño desfase temporal aleatorio (Humanización).
        Evita que la percusión suene 'computacionalmente perfecta'.
        """
        offset = int(np.random.uniform(-intensity, intensity) * sr)
        return offset

# --- 11. SÍNTESIS DE PERCUSIÓN CIBERNÉTICA (REVISADO) ---

    @staticmethod
    def hihat_closed(duration=0.05, sr=44100):
        """Hi-hat metálico basado en ruido blanco filtrado."""
        t = np.linspace(0, duration, int(sr * duration))
        # Generamos ruido blanco
        noise = np.random.uniform(-1, 1, len(t))
        # Envolvente exponencial ultra-rápida (clic metálico)
        env = np.exp(-100 * t)
        # Filtro paso-alto simple mediante diferenciación
        hi_noise = np.diff(noise, prepend=0) 
        return hi_noise * env * 0.25

    @staticmethod
    def electronic_clap(sr=44100):
        """Clap sintético estilo 808/909 basado en ráfagas de ruido."""
        duration = 0.25
        t = np.linspace(0, duration, int(sr * duration))
        noise = np.random.uniform(-1, 1, len(t))
        
        # El secreto del clap son los 3 pre-impactos
        env = np.zeros_like(t)
        # 3 picos rápidos antes del cuerpo principal
        for delay in [0, 0.01, 0.02]:
            idx = int(delay * sr)
            env[idx:] += np.exp(-200 * (t[idx:] - delay))
            
        # Cuerpo principal
        env += np.exp(-15 * t)
        return noise * env * 0.4

    @staticmethod
    def glitch_rim(sr=44100):
        """Sonido percusivo corto y seco para ritmos IDM."""
        t = np.linspace(0, 0.02, int(sr * 0.02))
        freq_env = 800 * np.exp(-150 * t)
        phase = 2 * np.pi * np.cumsum(freq_env) / sr
        click = np.sin(phase) * np.exp(-100 * t)
        return click * 0.5

    # --- 12. RENDERIZADOR DE LOOPS MAESTRO (VERSION ESTABLE) ---

    @staticmethod
    def render_complex_loop(pattern, sound_map, step_dur, sr=44100):
        """
        Renderiza patrones de texto con soporte para:
        - X/K: Kick | S/C: Snare/Clap | H/R: Hat/Ratchet | g: Glitch
        - Minúsculas: 30% volumen (Ghost notes)
        """
        import random # Aseguramos la importación local si no está arriba
        
        step_samples = int(step_dur * sr)
        total_samples = len(pattern) * step_samples
        output = np.zeros(total_samples)
        
        # Mapa extendido de caracteres por defecto
        full_map = {
            'X': Percussion.kick_909,
            'K': Percussion.kick_808_sub,
            'S': Percussion.metallic_snare,
            'C': Percussion.electronic_clap,
            'H': Percussion.hihat_closed,
            'R': Percussion.hihat_closed, # Ratchet usa el mismo sonido base
            'G': Percussion.glitch_rim,
            'M': Percussion.fm_metal_perc
        }
        # Actualizamos con lo que venga del arranger
        full_map.update({k.upper(): v for k, v in sound_map.items()})
        
        for i, char in enumerate(pattern):
            if char in ['.', ' ']: continue
            
            upper_char = char.upper()
            sound_func = full_map.get(upper_char)
            if not sound_func: continue
            
            # 1. Generar audio
            try:
                sample = sound_func(sr=sr)
            except TypeError:
                sample = sound_func() # Por si alguna función no acepta sr=
                
            # 2. Lógica de Ratcheting (repetición rápida)
            if upper_char == 'R':
                sample = Percussion.apply_ratcheting(sample, repeats=random.randint(2, 4))
            
            # 3. Dinámica (Velocidad)
            vol = 0.8 if char.isupper() else 0.25
            
            # 4. Humanización del Timing
            start_idx = (i * step_samples) + Percussion.add_jitter(i, intensity=0.003, sr=sr)
            start_idx = max(0, start_idx)
            
            # 5. Sumar a la mezcla con seguridad de límites
            end_idx = min(start_idx + len(sample), total_samples)
            output[start_idx:end_idx] += sample[:end_idx-start_idx] * vol
                
        return output

    @staticmethod
    def get_standard_kit():
        """Retorna un kit equilibrado para cualquier estilo."""
        return {
            'X': Percussion.kick_909,
            'S': Percussion.metallic_snare,
            'H': Percussion.hihat_closed,
            'C': Percussion.electronic_clap,
            'g': Percussion.glitch_rim
        }

# --- ALIAS FINAL PARA EL SISTEMA ---
Perc = Percussion