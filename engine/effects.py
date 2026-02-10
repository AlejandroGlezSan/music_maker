"""
ENGINE: EFFECTS v12.0 - MASTERING, DYNAMICS AND TEXTURE TOOLS
Improvements:
- Robust sidechain compressor with attack/release smoothing
- Riser shaping utility for transitions
- Vocoder shaper refined with envelope smoothing
- Transient shaper for percussion punch
- Stereo widener improved (mono input -> stereo output simulation)
- Safe, efficient implementations using numpy only
"""
import numpy as np

class Effects:
    @staticmethod
    def reverb_simple(data, room_size=0.5, damp=0.5, sr=44100):
        """
        Simple multi-tap reverb. Non-recursive, cheap.
        """
        if data.size == 0:
            return data
        delays = [int(0.029 * sr), int(0.037 * sr), int(0.044 * sr), int(0.050 * sr)]
        output = np.copy(data).astype(np.float32)
        for i, d in enumerate(delays):
            if d < len(data):
                decay = room_size * (0.6 ** i)
                reverb_layer = np.zeros_like(data)
                reverb_layer[d:] = data[:-d] * decay * (1.0 - damp)
                output += reverb_layer
        # gentle normalization
        maxv = np.max(np.abs(output)) if output.size else 1.0
        if maxv > 1.0:
            output = output / maxv
        return output

    @staticmethod
    def bitcrusher(data, bits=8):
        if data.size == 0:
            return data
        q = 2 ** (bits - 1)
        return np.round(data * q) / q

    @staticmethod
    def sidechain_compressor(data, trigger, threshold=0.1, ratio=4.0,
                             attack=0.001, release=0.05, sr=44100):
        """
        Time-domain sidechain compressor.
        - data: signal to be compressed (mono)
        - trigger: sidechain signal (mono)
        Returns compressed data (same shape).
        """
        if data.size == 0:
            return data
        # Envelope follower on trigger
        env = np.abs(trigger)
        # smoothing coefficients
        a_attack = np.exp(-1.0 / (attack * sr)) if attack > 0 else 0.0
        a_release = np.exp(-1.0 / (release * sr)) if release > 0 else 0.0
        smooth = np.zeros_like(env)
        g = 1.0
        for i in range(len(env)):
            if env[i] > smooth[i-1] if i > 0 else env[i]:
                smooth[i] = a_attack * (smooth[i-1] if i > 0 else 0.0) + (1 - a_attack) * env[i]
            else:
                smooth[i] = a_release * (smooth[i-1] if i > 0 else 0.0) + (1 - a_release) * env[i]
        # compute gain reduction
        gain = np.ones_like(smooth)
        eps = 1e-9
        for i in range(len(smooth)):
            if smooth[i] > threshold + eps:
                desired = threshold + (smooth[i] - threshold) / ratio
                gain[i] = desired / (smooth[i] + eps)
            else:
                gain[i] = 1.0
        # smooth gain to avoid zippering
        gain = np.convolve(gain, np.ones(int(sr*0.005))/max(1,int(sr*0.005)), mode='same')
        return data * gain

    @staticmethod
    def ping_pong_delay(data, delay_ms=300, feedback=0.4, sr=44100):
        if data.size == 0:
            return data
        d_samples = int((delay_ms / 1000.0) * sr)
        output = np.copy(data).astype(np.float32)
        # simple ping-pong simulated by alternating polarity and decaying feedback
        fb = feedback
        pos = d_samples
        while pos < len(data):
            seg = data[:len(data)-pos] * fb
            output[pos:pos+len(seg)] += seg
            fb *= 0.7
            pos += d_samples
            if fb < 0.01:
                break
        # normalize
        maxv = np.max(np.abs(output)) if output.size else 1.0
        if maxv > 1.0:
            output = output / maxv
        return output

    @staticmethod
    def master_limiter(data, ceiling=0.95):
        if data.size == 0:
            return data
        max_val = np.max(np.abs(data))
        if max_val > ceiling and max_val > 0:
            return (data / max_val) * ceiling
        return data

    @staticmethod
    def stereo_widener(data, width=1.5, sr=44100):
        """
        If input is mono, simulate stereo by creating a delayed and slightly detuned copy.
        If input is stereo (2D array), apply mid/side widening.
        """
        if data.size == 0:
            return data
        # If data is 2D (stereo), assume shape (2, N)
        if data.ndim == 2 and data.shape[0] == 2:
            mid = (data[0] + data[1]) * 0.5
            side = (data[0] - data[1]) * 0.5
            side *= width
            left = mid + side
            right = mid - side
            out = np.vstack([left, right])
            # normalize
            maxv = np.max(np.abs(out)) if out.size else 1.0
            if maxv > 1.0:
                out = out / maxv
            return out
        # Mono input
        delay_samples = int(0.012 * sr)  # 12ms Haas-ish
        delayed = np.zeros_like(data)
        if len(data) > delay_samples:
            delayed[delay_samples:] = data[:-delay_samples]
        # slight detune via resampling approximation: small amplitude modulation
        detune = 1.0 + (0.001 * (width - 1.0))
        left = data
        right = delayed * detune
        out_left = (left + right * (width - 1.0)) / width
        out_right = (right + left * (width - 1.0)) / width
        # stack stereo
        out = np.vstack([out_left, out_right])
        return out

    @staticmethod
    def cinematic_reverb(data, decay=0.8, density=0.7, sr=44100):
        if data.size == 0:
            return data
        output = np.copy(data).astype(np.float32)
        delays_ms = [149, 107, 73, 41]
        for d_ms in delays_ms:
            d = int((d_ms / 1000.0) * sr)
            if d < len(data):
                layer = np.zeros_like(data)
                for i in range(1, 4):
                    pos = d * i
                    if pos < len(data):
                        layer[pos:] += data[:-pos] * (decay ** i)
                output += layer * (density / 4.0)
        return Effects.master_limiter(output * 0.6)

    @staticmethod
    def vocoder_shaper(carrier, modulator, sensitivity=0.8, bands=64):
        """
        Lightweight vocoder-like amplitude shaping:
        - carrier: source signal (e.g., pad)
        - modulator: modulator signal (e.g., voice)
        Returns carrier shaped by smoothed envelope of modulator.
        """
        if carrier.size == 0 or modulator.size == 0:
            return carrier
        # envelope of modulator
        env = np.abs(modulator)
        # smooth envelope with moving average
        win = max(1, int(len(env) / (bands * 2)))
        kernel = np.ones(win) / win
        smooth_env = np.convolve(env, kernel, mode='same')
        # normalize
        maxv = np.max(smooth_env) if smooth_env.size else 1.0
        if maxv > 0:
            smooth_env = smooth_env / maxv
        # apply sensitivity and shape
        norm_env = (smooth_env ** 0.9) * sensitivity
        # ensure same length as carrier
        if len(norm_env) < len(carrier):
            reps = int(np.ceil(len(carrier) / len(norm_env)))
            norm_env = np.tile(norm_env, reps)[:len(carrier)]
        else:
            norm_env = norm_env[:len(carrier)]
        return carrier * norm_env

    @staticmethod
    def chorus(data, rate=1.2, depth=0.002, sr=44100):
        if data.size == 0:
            return data
        t = np.linspace(0, len(data)/sr, len(data))
        lfo = depth * np.sin(2 * np.pi * rate * t)
        indices = np.arange(len(data))
        mod_idx = np.clip((indices - (lfo * sr).astype(int)), 0, len(data)-1)
        return (data + data[mod_idx]) * 0.5

    @staticmethod
    def tube_warmth(data, drive=2.5, warmth=0.4):
        if data.size == 0:
            return data
        # asymmetric drive for subtle even-harmonic coloration
        driven = np.where(data > 0,
                         np.tanh(data * drive),
                         np.tanh(data * drive * (1 - warmth)))
        # gentle mix with original for parallel saturation feel
        return (data * (1 - warmth * 0.6)) + (driven * (warmth * 0.6))

    @staticmethod
    def adaptive_compressor(data, threshold=0.15, ratio=4.0, attack=0.01, sr=44100):
        if data.size == 0:
            return data
        output = np.zeros_like(data)
        gain = 1.0
        alpha = np.exp(-1.0 / (attack * sr)) if attack > 0 else 0.0
        for i in range(len(data)):
            env = abs(data[i])
            if env > threshold:
                target = threshold + (env - threshold) / ratio
                target_gain = target / (env + 1e-9)
            else:
                target_gain = 1.0
            gain = alpha * gain + (1 - alpha) * target_gain
            output[i] = data[i] * gain
        return output

    @staticmethod
    def tilt_eq(data, brightness=0.2):
        if data.size == 0:
            return data
        highs = np.diff(data, prepend=0)
        return data + highs * brightness

    @staticmethod
    def transient_shaper(data, attack_gain=1.5, sustain_gain=0.9, sr=44100):
        """
        Simple transient shaper: boost attack portion and reduce sustain.
        Works by detecting transient via high-pass envelope.
        """
        if data.size == 0:
            return data
        # high-pass-ish transient detector
        hp = np.abs(np.concatenate(([0], np.diff(data))))
        win = max(1, int(0.005 * sr))
        env = np.convolve(hp, np.ones(win)/win, mode='same')
        # normalize
        maxv = np.max(env) if env.size else 1.0
        if maxv > 0:
            env = env / maxv
        # shape: attack region gets boost, sustain reduced
        shaped = data * (sustain_gain + (attack_gain - sustain_gain) * env)
        return shaped

    @staticmethod
    def riser_shaper(duration_sec=4.0, sr=44100, start_freq=50, end_freq=8000, intensity=0.8):
        """
        Generate a riser buffer (mono) that sweeps from start_freq to end_freq.
        Useful to layer under transitions.
        """
        num = int(duration_sec * sr)
        if num <= 0:
            return np.array([], dtype=np.float32)
        t = np.linspace(0, duration_sec, num)
        # exponential frequency sweep
        freqs = start_freq * ( (end_freq / start_freq) ** (t / duration_sec) )
        phase = 2 * np.pi * np.cumsum(freqs) / sr
        noise = np.random.uniform(-1.0, 1.0, num) * 0.3
        tone = np.sin(phase) * (t / duration_sec)  # ramp amplitude
        out = (tone + noise * 0.5) * intensity
        # gentle lowpass to avoid harshness
        kernel = np.ones(int(0.01 * sr)) / max(1, int(0.01 * sr))
        out = np.convolve(out, kernel, mode='same')
        # normalize
        maxv = np.max(np.abs(out)) if out.size else 1.0
        if maxv > 0:
            out = out / maxv * intensity
        return out.astype(np.float32)

# Alias
Ef = Effects
