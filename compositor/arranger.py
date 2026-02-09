import numpy as np
import random
from engine.oscillators import Osc
from engine.percussion import Perc
from engine.effects import Ef
from compositor.scales import Scales

class Arranger:
    """
    COMPOSITOR: ARRANGER v13.0 - DYNAMIC STORYTELLER
    -----------------------------------------------
    Motor de composición basado en actos con gestión de tensión.
    """
    def __init__(self, sr=44100):
        self.sr = sr
        self.scales_engine = Scales()
        self.bpm = 128

def compose(self, style="techno", duration_sec=30, emotion="cyberpunk"):
        num_samples = int(self.sr * duration_sec)
        master_bus = np.zeros(num_samples, dtype=np.float32) # Forzamos tipo float32
        
        scale_freqs, progression = self.scales_engine.get_emotional_setup(emotion)
        step_dur = 60 / self.bpm / 4 
        
        sections = {
            "intro": (0, 0.25, 0.2),
            "build": (0.25, 0.5, 0.6),
            "drop": (0.5, 0.8, 1.0),
            "outro": (0.8, 1.0, 0.3)
        }

        for name, (start_p, end_p, tension) in sections.items():
            start_s = int(num_samples * start_p)
            end_s = int(num_samples * end_p)
            length = end_s - start_s
            
            if length <= 0: continue

            # --- GENERACIÓN PROTEGIDA ---
            # Forzamos que cada capa sea un array plano y tenga el tamaño exacto
            perc_layer = np.zeros(length)
            raw_perc = self._generate_drums(style, length, step_dur, tension)
            perc_layer[:min(len(raw_perc), length)] = raw_perc[:length]

            # Elegimos frecuencia de la progresión
            prog_idx = int(tension * (len(progression)-1))
            root_freq = progression[prog_idx][0]
            
            bass_layer = np.zeros(length)
            raw_bass = self._generate_bass(style, length, root_freq, tension)
            bass_layer[:min(len(raw_bass), length)] = raw_bass[:length]
            
            melody_layer = np.zeros(length)
            raw_melody = self._generate_melody(scale_freqs, length, step_dur, tension)
            melody_layer[:min(len(raw_melody), length)] = raw_melody[:length]

            # --- MIXING CON SEGURIDAD ---
            # El error "setting an array element with a sequence" suele venir de aquí
            # si alguna de estas capas no es un array plano de NumPy.
            combined = (perc_layer * 0.7) + (bass_layer * 0.5) + (melody_layer * 0.3)
            
            if name in ["build", "intro"]:
                combined = self._apply_transition_fx(combined, tension)

            # Inserción segura
            master_bus[start_s:end_s] = combined[:length]

        return Ef.master_limiter(master_bus)

def _generate_drums(self, style, length, step_dur, tension):
        """Genera percusión que evoluciona con la tensión."""
        # A mayor tensión, más probabilidad de 'Ratcheting' (redobles)
        complexity = 0.3 + (tension * 0.6)
        pattern = Perc.generate_idm_sequence(steps=16, complexity=complexity)
        
        # Si estamos en el Drop, forzamos un kick 4/4 sólido
        if tension > 0.8:
            pattern = "X.H.X.H.X.H.X.H."
            
        kit = Perc.get_standard_kit()
        return Perc.render_complex_loop(pattern, kit, step_dur, sr=self.sr)[:length]

def _generate_bass(self, style, length, freq, tension):
        """Genera líneas de bajo tipo Acid o Sub."""
        t = np.linspace(0, length/self.sr, length)
        if tension > 0.7:
            # Bajo Acid 303 si la tensión es alta
            return Osc.acid_303_style(t, freq, resonance=tension, env_mod=0.7)
        else:
            # Bajo senoidal/sub profundo para intros
            return Osc.sine(t, freq) * np.exp(-2 * t) # Bass Pluck

def _generate_melody(self, scale, length, step_dur, tension):
        # Aseguramos que motif sea una lista de frecuencias, no una secuencia de arrays
        motif = self.scales_engine.generate_motif(scale, length=16, jumps=(tension > 0.5))
        
        step_s = int(step_dur * self.sr)
        audio = np.zeros(length)
        
        for i, freq in enumerate(motif):
            start = i * step_s
            if start >= length: break
            
            dur = min(step_s, length - start)
            t_note = np.linspace(0, dur/self.sr, dur)
            
            # Forzamos que la salida del oscilador sea un array plano
            note_audio = np.array(Osc.deep_space_pad(t_note, freq)).flatten()
            
            env = np.sin(np.pi * np.linspace(0, 1, len(note_audio)))
            end_idx = start + len(note_audio)
            if end_idx <= length:
                audio[start:end_idx] += note_audio * env
                
        return audio

def _apply_transition_fx(self, audio, tension):
        """Añade Risers y Washouts de Reverb al final de las secciones."""
        # Aplicamos un filtro que se abre al final
        fade_len = int(len(audio) * 0.2)
        if fade_len > 0:
            # Simular un Riser de ruido blanco
            t_rise = np.linspace(0, fade_len/self.sr, fade_len)
            riser = np.random.normal(0, 0.1, fade_len) * np.linspace(0, 1, fade_len)
            audio[-fade_len:] += riser
        return audio

def log(self, msg):
        print(f"[ARRANGER] >> {msg}")