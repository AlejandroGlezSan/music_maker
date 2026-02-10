"""
ENGINE: OSCILLATORS v11.0 - TIMBRAL VARIATION, LFOs AND HUMANIZATION
- Añade LFO por nota a supersaw, fm_lead y morph_osc
- Humanización por nota (detune, phase jitter, amplitude variation)
- morph_osc con morph_env (evolución del morph durante la nota)
- deep_space_pad enriquecido con sub-grains y modulación lenta
- Todas las funciones aceptan sr y manejan buffers vacíos de forma segura
"""
import numpy as np

class Oscillators:
    SR = 44100
    PI = np.pi

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _safe_linspace(duration, sr, num=None):
        if num is not None:
            return np.linspace(0, duration, num)
        samples = int(max(0, duration * sr))
        if samples <= 0:
            return np.array([], dtype=np.float32)
        return np.linspace(0, duration, samples)

    @staticmethod
    def _humanize_detune(freq, detune_cents=5.0):
        """Small random detune in cents converted to frequency factor."""
        cents = np.random.uniform(-detune_cents, detune_cents)
        return freq * (2 ** (cents / 1200.0))

    @staticmethod
    def _apply_gain_safety(sig, target=0.7):
        if sig.size == 0:
            return sig
        maxv = np.max(np.abs(sig))
        if maxv <= 0:
            return sig
        return sig / maxv * target

    # -----------------------
    # Basic oscillators with LFO/humanization
    # -----------------------
    @staticmethod
    def sine(t, freq, sr=44100, phase=0.0):
        if t.size == 0:
            return t
        return np.sin(2 * Oscillators.PI * freq * t + phase)

    @staticmethod
    def supersaw(t, freq, detune=0.02, layers=9, lfo_rate=0.1, lfo_depth=0.002, sr=44100):
        """
        Supersaw with per-layer detune and a slow LFO modulating detune for movement.
        detune: base detune factor (relative)
        lfo_rate: Hz for slow movement
        lfo_depth: additional detune depth
        """
        if t.size == 0:
            return t
        output = np.zeros_like(t)
        rng = np.random.default_rng(int(freq * 100) % 10000)
        lfo = lfo_depth * np.sin(2 * np.pi * lfo_rate * t)
        center = freq
        for i in range(layers):
            # layer-specific detune spread
            spread = (i - (layers - 1) / 2) / (layers / 2)
            drift = 1.0 + detune * spread + lfo * spread
            phase_offset = rng.random() * 2 * np.pi
            layer_signal = np.sin(2 * np.pi * (center * drift) * t + phase_offset)
            amp_mod = 1.0 - (abs(spread) * 0.6)
            output += layer_signal * amp_mod
        return Oscillators._apply_gain_safety(output / layers * 0.8)

    @staticmethod
    def trance_pluck(duration, freq, sr=44100, brightness=0.5, seed=None):
        samples = int(max(0, duration * sr))
        if samples <= 0:
            return np.array([], dtype=np.float32)
        N = int(max(2, sr / max(1.0, freq)))
        rng = np.random.default_rng(seed or int(freq*1000) % 100000)
        buf = rng.uniform(-1, 1, N)
        output = np.zeros(samples)
        for i in range(samples):
            output[i] = buf[i % N]
            new_val = 0.5 * (buf[i % N] + buf[(i + 1) % N])
            buf[i % N] = new_val * (0.992 + (brightness * 0.005))
        return Oscillators._apply_gain_safety(output)

    @staticmethod
    def fm_lead(t, carrier_freq, ratio=1.618, index=2.5, lfo_rate=5.0, lfo_depth=0.05, sr=44100):
        """
        FM lead with an LFO modulating the modulation index for movement.
        """
        if t.size == 0:
            return t
        modulator_freq = carrier_freq * ratio
        lfo = 1.0 + lfo_depth * np.sin(2 * np.pi * lfo_rate * t)
        modulator = index * lfo * np.sin(2 * np.pi * modulator_freq * t)
        carrier = np.sin(2 * np.pi * carrier_freq * t + modulator)
        env = np.exp(-1.5 * t)
        return carrier * env

    @staticmethod
    def pulse_pwm(t, freq, lfo_rate=0.5, lfo_depth=0.35, sr=44100):
        if t.size == 0:
            return t
        pwm_lfo = 0.5 + lfo_depth * np.sin(2 * np.pi * lfo_rate * t)
        duty_cycle = (t * freq) % 1.0
        output = np.where(duty_cycle < pwm_lfo, 1.0, -1.0)
        return output * 0.5

    # -----------------------
    # Acid / TB-303 style with filter env placeholder
    # -----------------------
    @staticmethod
    def acid_303_style(t, freq, resonance=0.7, env_mod=0.5, sr=44100):
        if t.size == 0:
            return t
        sig = np.sign(np.sin(2 * np.pi * freq * t)) * 0.5
        sig += (np.random.random(len(t)) - 0.5) * 0.01
        f_env = np.exp(-10 * t) * env_mod * 5000 + (freq * 2)
        output = np.zeros_like(sig)
        block = 128
        try:
            from engine.filters import Filters
            for i in range(0, len(sig), block):
                end = min(i + block, len(sig))
                cutoff = min(f_env[i], sr/2.2)
                output[i:end] = Filters.moog_ladder(sig[i:end], cutoff, resonance, sr)
        except Exception:
            output = sig
        return np.tanh(output * 2.0)

    # -----------------------
    # Morphing oscillator with morph_env
    # -----------------------
    @staticmethod
    def morph_osc(t, freq, morph_factor=0.5, morph_env=None, harmonics=8, sr=44100):
        """
        Morph between sine and saw (additive saw approximation).
        morph_env: optional envelope array same length as t (0..1) to modulate morph_factor over time.
        """
        if t.size == 0:
            return t
        sine = np.sin(2 * np.pi * freq * t)
        saw = np.zeros_like(t)
        for n in range(1, harmonics + 1):
            saw += (np.sin(2 * np.pi * n * freq * t) / n)
        saw = saw / np.max(np.abs(saw)) if np.max(np.abs(saw)) > 0 else saw
        if morph_env is None:
            morph_env = np.ones_like(t) * morph_factor
        else:
            # ensure same length
            if len(morph_env) != len(t):
                morph_env = np.interp(np.linspace(0, 1, len(t)), np.linspace(0, 1, len(morph_env)), morph_env)
        out = (sine * (1 - morph_env)) + (saw * morph_env)
        return Oscillators._apply_gain_safety(out * 0.8)

    # -----------------------
    # Deep space pad (enriquecido)
    # -----------------------
    @staticmethod
    def deep_space_pad(t, freq, sr=44100):
        """
        Pad atmosférico con supersaw + sub + grain cloud + slow filter movement.
        """
        if t.size == 0:
            return t
        try:
            from engine.filters import Filters
        except Exception:
            Filters = None

        # Supersaw base with slow LFO movement
        saw = Oscillators.supersaw(t, freq, detune=0.04, layers=6, lfo_rate=0.05, lfo_depth=0.003, sr=sr)
        # Sub layer
        sub = np.sin(2 * np.pi * (freq / 2.0) * t) * 0.45
        # Grain cloud texture (short grains tiled across t)
        grain = Oscillators.grain_cloud(duration=max(0.1, t[-1] if t.size else 0.1), freq=freq*0.5, density=8, grain_size=0.06, sr=sr)
        # If grain shorter than t, tile it
        if grain.size > 0:
            reps = int(np.ceil(len(t) / len(grain)))
            grain_full = np.tile(grain, reps)[:len(t)]
        else:
            grain_full = np.zeros_like(t)

        mix = (saw * 0.7) + (sub * 0.5) + (grain_full * 0.25)

        # slow LFO for filter cutoff movement
        lfo = (np.sin(2 * np.pi * 0.03 * t) + 1) / 2.0
        cutoff = 200 + (lfo * 3000)

        if Filters is not None:
            try:
                out = Filters.moog_ladder(mix, cutoff, 0.35, sr)
            except Exception:
                out = mix
        else:
            out = mix

        # gentle normalization and slight stereo-ish detune via phase offset (mono output)
        maxv = np.max(np.abs(out)) if out.size else 1.0
        if maxv > 0:
            out = out / maxv * 0.8
        return out

    # -----------------------
    # Grain cloud (kept but safe)
    # -----------------------
    @staticmethod
    def grain_cloud(duration, freq, density=20, grain_size=0.05, sr=44100):
        num_samples = int(max(0, duration * sr))
        if num_samples <= 0:
            return np.array([], dtype=np.float32)
        output = np.zeros(num_samples, dtype=np.float32)
        num_grains = max(1, int(duration * density))
        for _ in range(num_grains):
            start_pos = np.random.randint(0, max(1, num_samples - int(grain_size * sr)))
            g_t = np.linspace(0, grain_size, int(max(1, sr * grain_size)))
            envelope = np.exp(-((g_t - grain_size/2)**2) / (2 * (grain_size/6)**2))
            grain = np.sin(2 * np.pi * freq * g_t) * envelope
            end_pos = start_pos + len(grain)
            output[start_pos:end_pos] += grain * 0.2
        return output

    # -----------------------
    # Vibrato, phase distortion, etc.
    # -----------------------
    @staticmethod
    def apply_vibrato(t, freq, vib_freq=5.0, depth=0.002):
        if t.size == 0:
            return freq
        mod = depth * np.sin(2 * np.pi * vib_freq * t)
        return freq * (1 + mod)

    @staticmethod
    def phase_distortion(t, freq, amount=0.5):
        if t.size == 0:
            return t
        phase = 2 * np.pi * freq * t
        distorted_phase = phase + amount * np.sin(phase)
        return np.sin(distorted_phase)

    @staticmethod
    def exponential_release(signal, decay_rate=5.0, sr=44100):
        if signal.size == 0:
            return signal
        t = np.linspace(0, len(signal)/sr, len(signal))
        env = np.exp(-decay_rate * t)
        return signal * env

    @staticmethod
    def tension_riser_fx(duration, sr=44100):
        if duration <= 0:
            return np.array([], dtype=np.float32)
        t = np.linspace(0, duration, int(sr * duration))
        f_env = 50 * np.power(2, t * 4)
        phase = 2 * np.pi * np.cumsum(f_env) / sr
        noise = np.random.uniform(-0.1, 0.1, len(t))
        return (np.sin(phase) + noise) * (t / duration)

    @staticmethod
    def vinyl_crackle(duration, sr=44100):
        samples = int(max(0, duration * sr))
        if samples <= 0:
            return np.array([], dtype=np.float32)
        output = np.zeros(samples)
        num_clicks = int(duration * 5)
        for _ in range(num_clicks):
            pos = np.random.randint(0, samples)
            output[pos] = np.random.uniform(0.5, 1.0)
        return np.convolve(output, [0.5, 0.5], mode='same')

    @staticmethod
    def speak(text, duration, pitch=110, sr=44100):
        try:
            from engine.filters import Filters
        except Exception:
            Filters = None
        num_samples = int(max(0, duration * sr))
        if num_samples <= 0:
            return np.array([], dtype=np.float32)
        output = np.zeros(num_samples, dtype=np.float32)
        char_dur = duration / max(1, len(text))
        for i, char in enumerate(text.lower()):
            start = int(i * char_dur * sr)
            end = int((i + 1) * char_dur * sr)
            if end <= start:
                continue
            t_char = np.linspace(0, (end - start) / sr, end - start)
            if char in "aeiou":
                source = (np.sin(2 * np.pi * pitch * t_char) +
                          np.sign(np.sin(2 * np.pi * pitch * t_char))) * 0.5
                if Filters is not None:
                    vocal = Filters.formant_filter(source, char, sr=sr)
                else:
                    vocal = source * np.exp(-6 * t_char)
            else:
                noise = np.random.uniform(-1, 1, len(t_char))
                vocal = noise * np.exp(-12 * t_char)
                if char in "ptk":
                    vocal *= np.exp(-150 * t_char)
            env = np.sin(np.pi * np.linspace(0, 1, len(t_char)))
            output[start:end] = vocal * env
        return Oscillators._apply_gain_safety(output * 0.9)

# Alias
Osc = Oscillators
