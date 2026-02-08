import numpy as np
import random
from engine.oscillators import Oscillators as Osc
from engine.percussion import Percussion as Perc
from engine.effects import Effects as Ef
from engine.filters import Filters as Fil

class Arranger:
    def __init__(self, sr=44100):
        self.sr = sr
        from compositor.sequencer import Sequencer
        self.seq = Sequencer(sr)

    def compose(self, style, duration_sec):
        from compositor.scales import CHORDS, PROGRESSIONS
        
        # Configuración Dubstep: 140 BPM es el estándar
        self.bpm = 140 if style in ["dubstep", "brostep"] else random.randint(124, 130)
        self.current_prog = random.choice(PROGRESSIONS)
        step_dur = (60 / self.bpm) / 4
        bar_len = int(self.sr * step_dur * 16)
        total_bars = max(1, int(duration_sec / (step_dur * 16)))
        
        full_song = []
        for bar_idx in range(total_bars):
            progress = bar_idx / total_bars
            section = "drop" if 0.4 < progress < 0.8 else "build"
            full_song.append(self._generate_unique_bar(style, section, step_dur, bar_idx, total_bars, bar_len))
            
        return np.concatenate(full_song)

    def _generate_unique_bar(self, style, section, step_dur, bar_idx, total_bars, bar_len):
        from compositor.scales import CHORDS
        pats = self.seq.get_default_patterns(style)
        chord_name = self.current_prog[bar_idx % len(self.current_prog)]
        f_root = CHORDS[chord_name][0]
        
        # 1. PERCUSIÓN (Heavy Snare para Brostep)
        t_kick = self.seq.create_track(pats['k'], lambda: Perc.kick(self.sr, "techno"), step_dur, bar_len)
        # Usamos una mezcla de kick y ruido para un snare potente
        t_snare = self.seq.create_track(pats.get('s', "....X..."), lambda: Perc.snare(self.sr) * 1.5, step_dur, bar_len)
        drums = t_kick + t_snare

        # 2. BASS ENGINE (La complejidad del estilo)
        t_bar = np.linspace(0, step_dur * 16, bar_len, endpoint=False)
        bass_layer = np.zeros(bar_len)
        
        if section == "drop":
            if style == "dubstep":
                # Dubstep Clásico: Wobble profundo de 1/4 o 1/8
                lfo_speed = random.choice([2, 4, 8]) 
                bass_layer = Osc.wobble_bass(t_bar, f_root / 2, lfo_speed) * 0.4
            elif style == "brostep":
                # Brostep: Cambios rápidos de LFO (Glitchy) y Growls
                # Dividimos el compás en fragmentos de LFO aleatorios
                intensity = np.sin(np.linspace(0, np.pi, bar_len)) * 5
                bass_layer = Osc.brostep_growl(t_bar, f_root / 2, intensity) * 0.3
        
        # 3. ATMÓSFERA
        synths = np.zeros(bar_len)
        if section == "build":
            for f in CHORDS[chord_name]:
                synths += Ef.stereo_widener(Osc.supersaw(t_bar, f, detune=0.02) * 0.05)

        # 4. SUB-BASS (Vital en Dubstep)
        sub = Ef.sidechain(Osc.sub_bass(t_bar, f_root / 2) * 0.5, pats['k'], step_dur, self.sr)

        # Finalización
        synths = Fil.resonant_low_pass(synths + bass_layer, 2000 if section == "drop" else 5000, 0.5, self.sr)
        synths = Ef.sidechain(synths, pats['k'], step_dur, self.sr)

        return Ef.master_limiter(drums + synths + sub)