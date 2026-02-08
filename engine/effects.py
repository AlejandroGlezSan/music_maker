import numpy as np

class Effects:
    @staticmethod
    def sidechain(data, kick_pattern, step_dur, sr=44100):
        samples_per_step = int(sr * step_dur)
        envelope = np.ones(len(data))
        for i, char in enumerate(kick_pattern):
            if char.upper() == "X":
                start = i * samples_per_step
                end = start + int(samples_per_step * 0.9)
                if end < len(envelope):
                    envelope[start:end] *= np.linspace(0.0, 1.0, end - start)
        return data * envelope

    @staticmethod
    def apply_delay(data, delay_ms, feedback, sr=44100):
        delay_samples = int((delay_ms / 1000) * sr)
        output = np.copy(data)
        if delay_samples < len(data):
            for i in range(delay_samples, len(data)):
                output[i] += output[i - delay_samples] * feedback
        return output

    @staticmethod
    def stereo_widener(data, shift_samples=15):
        # Micro-delay para ensanchar el sonido
        shifted = np.roll(data, shift_samples)
        return (data + shifted) * 0.5

    @staticmethod
    def master_limiter(data):
        max_val = np.max(np.abs(data))
        if max_val > 0.95:
            return data * (0.95 / max_val)
        return data