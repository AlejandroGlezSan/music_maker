"""
ENGINE: PERCUSSION v11.3 - GENERATIVE IDM & ANALOG MODELING
Enhancements:
- Probabilistic ratcheting in fills
- Polyrhythmic hi-hat generator
- Micro-swing per-step and section-aware swing
- Ghost notes 'x' handling (lower velocity + shortened)
- Automatic fills at bar ends
"""
import numpy as np
import random
import math

class Percussion:
    SR = 44100

    # -----------------------
    # Basic instruments
    # -----------------------
    @staticmethod
    def generate_idm_sequence(steps=16, complexity=0.6):
        """
        Crea un patrón de strings donde:
        'X' = Golpe
        '.' = Silencio
        'R' = Ratchet
        'g' = Ghost note
        '?' = Glitch
        Garantiza al menos un 'X' si complexity > 0.1.
        """
        import random
        pattern = []
        for i in range(steps):
            r = random.random()
            if r < complexity * 0.1:
                pattern.append('?')
            elif r < complexity * 0.2:
                pattern.append('R')
            elif r < complexity * 0.4:
                pattern.append('g')
            elif r < complexity:
                pattern.append('X')
            else:
                pattern.append('.')

        if 'X' not in pattern and complexity > 0.1:
            idx = random.randrange(0, steps)
            pattern[idx] = 'X'

        return pattern

    @staticmethod
    def kick_909(sr=44100, punch=1.2, decay=0.3):
        t = np.linspace(0, decay, int(sr * decay))
        f_env = 150 * np.exp(-40 * t * punch) + 45
        phase = 2 * np.pi * np.cumsum(f_env) / sr
        amp_env = np.exp(-6 * t)
        return np.tanh(np.sin(phase) * amp_env * 1.5)

    @staticmethod
    def kick_808_sub(sr=44100, decay=0.8, freq=40):
        t = np.linspace(0, decay, int(sr * decay))
        f_env = 100 * np.exp(-25 * t) + freq
        phase = 2 * np.pi * np.cumsum(f_env) / sr
        amp_env = np.exp(-3 * t)
        return np.sin(phase) * amp_env

    @staticmethod
    def metallic_snare(duration=0.15, resonance=0.9):
        t = np.linspace(0, duration, int(Percussion.SR * duration))
        mod = np.sin(2 * np.pi * 1200 * t) * resonance
        carrier = np.sin(2 * np.pi * 220 * t + mod)
        noise = np.random.uniform(-0.5, 0.5, len(t)) * np.exp(-40 * t)
        return (carrier + noise) * np.exp(-20 * t)

    @staticmethod
    def snare_ghost(sr=44100):
        t = np.linspace(0, 0.05, int(sr * 0.05))
        noise = np.random.uniform(-0.3, 0.3, len(t))
        return noise * np.exp(-100 * t)

    @staticmethod
    def hihat_closed(sr=44100):
        t = np.linspace(0, 0.03, int(sr * 0.03))
        noise = np.random.uniform(-1, 1, len(t))
        return noise * np.exp(-120 * t) * 0.4

    @staticmethod
    def hihat_fm(sr=44100, mod_index=5):
        t = np.linspace(0, 0.1, int(sr * 0.1))
        mod = np.sin(2 * np.pi * 5000 * t) * mod_index
        carrier = np.sin(2 * np.pi * 8000 * t + mod)
        return carrier * np.exp(-50 * t) * 0.2

    @staticmethod
    def glitch_click(duration=0.01):
        t = np.linspace(0, duration, int(Percussion.SR * duration))
        return np.random.uniform(-1, 1, len(t)) * np.exp(-200 * t)

    @staticmethod
    def glitch_burst(sr=44100):
        dur = np.random.uniform(0.01, 0.05)
        t = np.linspace(0, dur, int(sr * dur))
        freq = np.random.uniform(1000, 8000)
        sig = np.sin(2 * np.pi * freq * t) * np.random.choice([0, 1], len(t), p=[0.1, 0.9])
        return sig * 0.3

    # -----------------------
    # Utility: Euclidean (Bjorklund)
    # -----------------------
    @staticmethod
    def _euclidean(steps, pulses):
        if pulses <= 0 or steps <= 0:
            return [0] * steps
        if pulses >= steps:
            return [1] * steps

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
        return seq[:steps]

    # -----------------------
    # Polyrhythmic hi-hat generator
    # -----------------------
    @staticmethod
    def generate_polyrhythm_hihat(steps_main=16, steps_poly=12, pulses_poly=4):
        """
        Map a euclidean poly pattern onto a main grid.
        Returns a string of length steps_main with 'X' and '.' and occasional 'x' ghosts.
        """
        main = ['.' for _ in range(steps_main)]
        poly = Percussion._euclidean(steps_poly, pulses_poly)
        for i, val in enumerate(poly):
            if val == 1:
                pos = int(round(i * (steps_main / steps_poly))) % steps_main
                main[pos] = 'X'
        # add sparse ghost notes to fill texture
        for i in range(steps_main):
            if main[i] == '.' and random.random() < 0.06:
                main[i] = 'x'
        return "".join(main)

    # -----------------------
    # Core render loop (enhanced)
    # -----------------------
    @staticmethod
    def render_complex_loop(pattern, sound_map, step_dur, sr=44100,
                            swing=0.0, style='generic', steps_per_bar=16):
        """
        Renderiza el patrón con soporte para:
        - swing (0.0..0.5)
        - style-aware polyrhythms (e.g., techno hi-hat)
        - probabilistic ratcheting near bar ends
        - ghost notes 'x' (lower velocity + shortened)
        - fills auto-inserted at bar ends
        """
        step_samples = int(step_dur * sr)
        total_samples = len(pattern) * step_samples
        output = np.zeros(total_samples, dtype=np.float32)

        # Default mapping
        full_map = {
            'X': Percussion.kick_909,
            'K': Percussion.kick_808_sub,
            'S': Percussion.metallic_snare,
            'C': Percussion.electronic_clap if hasattr(Percussion, 'electronic_clap') else Percussion.metallic_snare,
            'H': Percussion.hihat_closed,
            'R': Percussion.hihat_closed,
            'G': Percussion.glitch_click,
            'M': Percussion.fm_metal_perc if hasattr(Percussion, 'fm_metal_perc') else Percussion.glitch_burst
        }
        # Merge user-provided map (case-insensitive)
        full_map.update({k.upper(): v for k, v in sound_map.items()})

        # If style requests polyrhythmic hats, pre-generate and merge
        if style == 'techno' and 'h' in sound_map:
            # create a polyrhythmic hat pattern and overlay onto pattern
            poly_hat = Percussion.generate_polyrhythm_hihat(steps_main=len(pattern), steps_poly=12, pulses_poly=5)
            # overlay: where poly_hat has 'X' and pattern has '.' we set 'H'
            pattern = list(pattern)
            for i, ch in enumerate(poly_hat):
                if ch == 'X' and pattern[i] in ['.', ' ']:
                    pattern[i] = 'H'
                elif ch == 'x' and pattern[i] in ['.', ' ']:
                    pattern[i] = 'x'
            pattern = "".join(pattern)

        for i, char in enumerate(pattern):
            if char in ['.', ' ']:
                continue

            upper_char = char.upper()
            sound_func = full_map.get(upper_char)
            if not sound_func:
                continue

            # Generate sample (try sr kw)
            try:
                sample = sound_func(sr=sr)
            except TypeError:
                sample = sound_func()

            # Ghost note handling: 'x' -> lower velocity, shortened sample
            is_ghost = (char == 'x')
            if is_ghost:
                vol = 0.35
                # shorten to make it feel like a ghost
                cut_len = max(1, int(len(sample) * 0.55))
                sample = sample[:cut_len]
            else:
                vol = 0.8 if char.isupper() else 0.5

            # Determine if we should ratchet here (higher chance near bar end)
            step_in_bar = i % steps_per_bar
            ratchet_chance = 0.02  # base
            # increase chance in last 2 steps of bar and in builds
            if step_in_bar >= steps_per_bar - 2:
                ratchet_chance += 0.25
            if style == 'idm':
                ratchet_chance += 0.05
            if random.random() < ratchet_chance:
                # apply ratcheting with variable repeats
                repeats = random.randint(2, 6)
                sample = Percussion.apply_ratcheting(sample, repeats=repeats)
                # ratcheting reduces overall level a bit
                vol *= 0.9

            # Micro-swing: small random offset around swing value
            swing_offset = 0
            if swing and (i % 2 != 0):
                micro = swing * (0.8 + 0.4 * random.random())
                swing_offset = int(step_samples * micro)

            # Add small jitter to avoid robotic placement
            jitter = Percussion.add_jitter(i, intensity=0.0015, sr=sr)

            start_idx = (i * step_samples) + swing_offset + jitter
            start_idx = max(0, start_idx)
            end_idx = min(start_idx + len(sample), total_samples)

            # Mix into output
            if start_idx < total_samples:
                output[start_idx:end_idx] += sample[:end_idx - start_idx] * vol

        # Automatic fills: if last bar is too sparse, add a fill
        bars = max(1, len(pattern) // steps_per_bar)
        for b in range(bars):
            bar_start = b * steps_per_bar * step_samples
            bar_end = min((b + 1) * steps_per_bar * step_samples, total_samples)
            bar_seg = output[bar_start:bar_end]
            # if energy is low, add a short glitch burst or ratcheted snare at the end
            if np.max(np.abs(bar_seg)) < 0.02 and random.random() < 0.35:
                # choose fill type
                if random.random() < 0.5:
                    fill = Percussion.glitch_burst(sr=sr) * 0.6
                else:
                    fill = Percussion.metallic_snare(duration=0.08) * 0.9
                    fill = Percussion.apply_ratcheting(fill, repeats=random.randint(3, 6))
                pos = bar_end - len(fill)
                if pos < bar_start:
                    pos = bar_start
                pos = int(pos)
                end = min(pos + len(fill), total_samples)
                output[pos:end] += fill[:end - pos] * 0.9

        # Safety: if output is silent, fill with textured hi-hat
        if np.max(np.abs(output)) < 1e-6:
            hh = Percussion.hihat_closed(sr=sr)
            if hh.size > 0:
                reps = int(np.ceil(total_samples / len(hh)))
                output = np.tile(hh, reps)[:total_samples] * 0.3

        return output

    # -----------------------
    # FM / metallic perc
    # -----------------------
    @staticmethod
    def fm_metal_perc(sr=44100, duration=0.1):
        t = np.linspace(0, duration, int(sr * duration))
        mod = np.sin(2 * np.pi * 831.4 * t) * 8.0
        carrier = np.sin(2 * np.pi * 314.15 * t + mod)
        return carrier * np.exp(-35 * t) * 0.3

    @staticmethod
    def alien_bongo(sr=44100):
        t = np.linspace(0, 0.2, int(sr * 0.2))
        f_env = 400 * np.exp(-40 * t) + 60
        phase = 2 * np.pi * np.cumsum(f_env) / sr
        return np.sin(phase) * np.exp(-12 * t)

    @staticmethod
    def circuit_bend_short(sr=44100):
        t = np.linspace(0, 0.05, int(sr * 0.05))
        sig = np.random.choice([-1, 1], len(t)) * np.sin(2 * np.pi * 2000 * t)
        return sig * np.exp(-100 * t) * 0.2

    # -----------------------
    # Ratcheting and helpers
    # -----------------------
    @staticmethod
    def apply_ratcheting(audio_segment, repeats=2):
        n = len(audio_segment)
        if n < 100 or repeats <= 1:
            return audio_segment
        chunk_size = max(1, n // repeats)
        chunk = audio_segment[:chunk_size]
        env = np.linspace(1, 0, chunk_size)
        return np.tile(chunk * env, repeats)

    @staticmethod
    def add_jitter(step_index, intensity=0.005, sr=44100):
        offset = int(np.random.uniform(-intensity, intensity) * sr)
        return offset

    # -----------------------
    # Utility kit & presets
    # -----------------------
    @staticmethod
    def electronic_clap(sr=44100):
        duration = 0.25
        t = np.linspace(0, duration, int(sr * duration))
        noise = np.random.uniform(-1, 1, len(t))
        env = np.zeros_like(t)
        for delay in [0, 0.01, 0.02]:
            idx = int(delay * sr)
            env[idx:] += np.exp(-200 * (t[idx:] - delay))
        env += np.exp(-15 * t)
        return noise * env * 0.4

    @staticmethod
    def glitch_rim(sr=44100):
        t = np.linspace(0, 0.02, int(sr * 0.02))
        freq_env = 800 * np.exp(-150 * t)
        phase = 2 * np.pi * np.cumsum(freq_env) / sr
        click = np.sin(phase) * np.exp(-100 * t)
        return click * 0.5

    @staticmethod
    def get_standard_kit():
        return {
            'X': Percussion.kick_909,
            'S': Percussion.metallic_snare,
            'H': Percussion.hihat_closed,
            'C': Percussion.electronic_clap,
            'g': Percussion.glitch_rim
        }

# Alias
Perc = Percussion
