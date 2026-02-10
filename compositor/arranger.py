"""
COMPOSITOR: ARRANGER v15.0 - LAYERED STORYTELLER WITH SIDECHAIN & TRANSITIONS
- Sidechain ducking (kick -> pads/arp)
- Section risers and sweep automation
- Vocoder texture insertion
- Smarter mixing weights per section to avoid isolated percussion
"""
import numpy as np
import random
from engine.oscillators import Osc
from engine.percussion import Perc
from engine.effects import Ef
from compositor.scales import Scales

class Arranger:
    def __init__(self, sr=44100):
        self.sr = sr
        self.scales_engine = Scales()
        self.bpm = 128

    def compose(self, style="techno", duration_sec=30, emotion="cyberpunk"):
        num_samples = int(self.sr * duration_sec)
        master_bus = np.zeros(num_samples, dtype=np.float32)

        scale_freqs, progression = self.scales_engine.get_emotional_setup(emotion)
        step_dur = 60.0 / self.bpm / 4.0

        # Continuous supportive layers
        pad_layer = self._generate_pad(scale_freqs, num_samples, duration_sec)
        bass_layer = self._generate_continuous_bass(scale_freqs, num_samples)
        arp_layer = self._generate_arp(scale_freqs, num_samples, step_dur, duration_sec)

        # Optional voice texture (sparse)
        voice_texture = self._generate_voice_texture(duration_sec, num_samples) * 0.0

        # Section map
        sections = {
            "intro": (0.0, 0.25, 0.2),
            "build": (0.25, 0.5, 0.6),
            "drop":  (0.5, 0.8, 1.0),
            "outro": (0.8, 1.0, 0.3)
        }

        for name, (start_p, end_p, tension) in sections.items():
            start_s = int(num_samples * start_p)
            end_s = int(num_samples * end_p)
            length = end_s - start_s
            if length <= 0:
                continue

            # Percussion loop for the section
            perc_loop = self._generate_drums(style, length, step_dur, tension)

            # Ensure percussion covers section
            if len(perc_loop) < length and len(perc_loop) > 0:
                reps = int(np.ceil(length / len(perc_loop)))
                perc_loop = np.tile(perc_loop, reps)[:length]

            # Section stabs and risers
            stab_layer = self._generate_stabs(scale_freqs, length, tension)
            riser = self._generate_riser(length, tension)

            # Extract slices from continuous layers
            pad_slice = pad_layer[start_s:end_s]
            bass_slice = bass_layer[start_s:end_s]
            arp_slice = arp_layer[start_s:end_s]
            voice_slice = voice_texture[start_s:end_s]

            # Apply sidechain: duck pads/arp by kick envelope
            kick_env = self._extract_kick_envelope(perc_loop, sr=self.sr)
            pad_sc = Ef.sidechain_compressor(pad_slice, kick_env, threshold=0.02, ratio=6)
            arp_sc = Ef.sidechain_compressor(arp_slice, kick_env, threshold=0.02, ratio=4)

            # Section-dependent gains
            perc_gain = 0.9
            bass_gain = 0.5 + (tension * 0.6)
            pad_gain = 0.35 + (0.4 * (1 - tension))
            arp_gain = 0.12 + (tension * 0.6)
            stab_gain = 0.5 * tension
            riser_gain = 0.6 * (tension if name == "build" else (tension * 0.3))

            combined = (
                (perc_loop * perc_gain) +
                (bass_slice * bass_gain) +
                (pad_sc * pad_gain) +
                (arp_sc * arp_gain) +
                (stab_layer * stab_gain) +
                (riser * riser_gain) +
                (voice_slice * 0.25)
            )

            # Apply section filter automation for movement
            combined = self._apply_filter_movement(combined, tension, name)

            # Crossfade edges
            combined = self._apply_section_fades(combined)

            # Safety fill if near-silent
            if np.max(np.abs(combined)) < 1e-5:
                hh = Perc.hihat_closed(sr=self.sr)
                if hh.size > 0:
                    reps = int(np.ceil(length / len(hh)))
                    combined = np.tile(hh, reps)[:length] * 0.35

            master_bus[start_s:end_s] = combined[:length]

        # Final processing chain
        master_bus = Ef.tube_warmth(master_bus, drive=1.0 + 0.2, warmth=0.18)
        master_bus = Ef.adaptive_compressor(master_bus, threshold=0.12, ratio=3.0, attack=0.005, sr=self.sr)
        master_bus = Ef.master_limiter(master_bus, ceiling=0.95)
        return master_bus

    # -----------------------
    # Layer generators
    # -----------------------
    def _generate_pad(self, scale_freqs, num_samples, duration_sec):
        t = np.linspace(0, duration_sec, num_samples)
        root_freq = scale_freqs[0] if len(scale_freqs) > 0 else 110.0
        try:
            pad = Osc.deep_space_pad(t, root_freq, sr=self.sr)
        except Exception:
            pad = Osc.supersaw(t, root_freq, detune=0.03, layers=6)
        # LFO and slow filter movement
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.03 * t)
        pad = pad * lfo * 0.5
        # gentle smoothing
        kernel = np.ones(2048) / 2048
        pad = np.convolve(pad, kernel, mode='same')
        return pad.astype(np.float32)

    def _generate_continuous_bass(self, scale_freqs, num_samples):
        t = np.linspace(0, num_samples / self.sr, num_samples)
        tonic = scale_freqs[0] if len(scale_freqs) > 0 else 55.0
        mod = 1.0 + 0.03 * np.sin(2 * np.pi * 0.08 * t)
        bass = np.sin(2 * np.pi * (tonic / 2) * mod * t) * 0.6
        burst_env = (np.sin(2 * np.pi * 0.25 * t) > 0).astype(float) * 0.12
        bass += burst_env * 0.18 * np.sin(2 * np.pi * tonic * t)
        # lowpass-ish smoothing via convolution
        kernel = np.ones(512) / 512
        bass = np.convolve(bass, kernel, mode='same')
        return bass.astype(np.float32)

    def _generate_arp(self, scale_freqs, num_samples, step_dur, duration_sec):
        if len(scale_freqs) == 0:
            return np.zeros(num_samples, dtype=np.float32)
        motif = self.scales_engine.generate_motif(scale_freqs, length=32, jumps=True)
        step_s = int(step_dur * self.sr / 2)
        arp = np.zeros(num_samples, dtype=np.float32)
        for i, freq in enumerate(motif):
            start = i * step_s
            if start >= num_samples:
                break
            dur = min(step_s, num_samples - start)
            t_note = np.linspace(0, dur / self.sr, dur)
            morph = 0.2 + 0.8 * ((i % 4) / 3.0)
            note = Osc.morph_osc(t_note, freq, morph) * np.exp(-6 * t_note)
            arp[start:start+len(note)] += note * (0.6 / (1 + (i % 4)))
        arp = Ef.chorus(arp, rate=0.9, depth=0.001, sr=self.sr)
        return arp

    def _generate_stabs(self, scale_freqs, length, tension):
        if len(scale_freqs) < 3:
            return np.zeros(length, dtype=np.float32)
        idx = 0 if tension < 0.6 else 3
        freqs = [
            scale_freqs[idx % len(scale_freqs)],
            scale_freqs[(idx + 2) % len(scale_freqs)],
            scale_freqs[(idx + 4) % len(scale_freqs)]
        ]
        t = np.linspace(0, length / self.sr, length)
        chord = np.zeros(length)
        for f in freqs:
            chord += np.sin(2 * np.pi * f * t) * np.exp(-8 * t)
        env = np.sin(np.pi * np.linspace(0, 1, length)) ** 0.8
        stab = chord * env * (0.35 + tension * 0.8)
        if tension < 0.4:
            stab *= 0.5
        # make stabs sparse: keep only a few transient bursts
        mask = np.zeros(length)
        step = max(1, int(self.sr * 0.25))
        for i in range(0, length, step):
            if random.random() < (0.25 + tension * 0.5):
                mask[i:i+int(self.sr*0.06)] = 1.0
        return (stab * mask).astype(np.float32)

    def _generate_riser(self, length, tension):
        # riser present mainly in build/drop; short and subtle otherwise
        dur = max(1, int(self.sr * 0.5))
        t = np.linspace(0, dur / self.sr, dur)
        riser = Osc.tension_riser_fx(dur / self.sr, sr=self.sr)
        # tile or trim to section length
        if len(riser) < length:
            reps = int(np.ceil(length / len(riser)))
            riser = np.tile(riser, reps)[:length]
        else:
            riser = riser[:length]
        # scale by tension
        return riser * (0.2 + tension * 0.8)

    def _generate_drums(self, style, length, step_dur, tension):
        complexity = 0.35 + (tension * 0.55)
        pattern = Perc.generate_idm_sequence(steps=16, complexity=complexity)
        if tension > 0.8:
            pattern = list("X...X...X...X...")
        kit = Perc.get_standard_kit()
        loop = Perc.render_complex_loop(pattern, kit, step_dur, sr=self.sr, swing=0.02 + (tension*0.03), style=style, steps_per_bar=16)
        if loop.size == 0:
            hh = Perc.hihat_closed(sr=self.sr)
            loop = np.tile(hh, int(np.ceil(length / len(hh))))[:length]
        return loop[:length]

    # -----------------------
    # Utilities: sidechain, filters, fades
    # -----------------------
    def _extract_kick_envelope(self, perc_buffer, sr=44100):
        # crude envelope: absolute + lowpass smoothing
        env = np.abs(perc_buffer)
        kernel = np.ones(int(0.01 * sr)) / max(1, int(0.01 * sr))
        smooth = np.convolve(env, kernel, mode='same')
        # normalize
        maxv = np.max(smooth) if smooth.size else 1.0
        if maxv > 0:
            smooth = smooth / maxv
        return smooth

    def _apply_filter_movement(self, audio, tension, section_name):
        # simple tilt: higher tension -> brighter
        brightness = 0.2 + (tension * 0.6)
        return Ef.tilt_eq(audio, brightness=brightness)

    def _apply_section_fades(self, audio):
        n = len(audio)
        if n <= 0:
            return audio
        fade = int(min(0.02 * self.sr, n * 0.05))
        if fade <= 0:
            return audio
        env = np.ones(n)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        return audio * env

    def _generate_voice_texture(self, duration_sec, num_samples):
        # sparse robotic phrase used as texture; low level by default
        try:
            phrase = "we are the machines"
            voice = Osc.speak(phrase, duration_sec * 0.25, pitch=160, sr=self.sr)
            # tile to full length but keep very low level
            reps = int(np.ceil(num_samples / len(voice))) if len(voice) > 0 else 1
            vt = np.tile(voice, reps)[:num_samples] * 0.08
            # apply vocoder shaping with pad (if pad exists later)
            return vt
        except Exception:
            return np.zeros(num_samples, dtype=np.float32)

    def _log(self, msg):
        print(f"[ARRANGER] >> {msg}")
