"""
SCALES ENGINE v9.0 - Theoretical Framework
------------------------------------------
Comprehensive music theory engine for algorithmic composition.
"""
import numpy as np

class Scales:
    REFERENCE_A4 = 440.0

    NOTE_MAP = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }

    MODES = {
        'ionian':     [0, 2, 4, 5, 7, 9, 11],
        'dorian':     [0, 2, 3, 5, 7, 9, 10],
        'phrygian':   [0, 1, 3, 5, 7, 8, 10],
        'lydian':     [0, 2, 4, 6, 7, 9, 11],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
        'aeolian':    [0, 2, 3, 5, 7, 8, 10],
        'locrian':    [0, 1, 3, 5, 6, 8, 10],
        'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
        'melodic_minor':  [0, 2, 3, 5, 7, 9, 11],
        'double_harmonic': [0, 1, 4, 5, 7, 8, 11],
        'phrygian_dominant': [0, 1, 4, 5, 7, 8, 10]
    }

    CHORDS_LIB = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'dim':   [0, 3, 6],
        'aug':   [0, 4, 8],
        'maj7':  [0, 4, 7, 11],
        'min7':  [0, 3, 7, 10],
        'dom7':  [0, 4, 7, 10],
        'm7b5':  [0, 3, 6, 10],
        'sus2':  [0, 2, 7],
        'sus4':  [0, 5, 7],
        'add9':  [0, 4, 7, 14],
        'min9':  [0, 3, 7, 10, 14],
        'dim7':  [0, 3, 6, 9]
    }

    def __init__(self, root_note='A', octave=4):
        self.root = root_note
        self.octave = octave

    def get_note_freq(self, note_name, octave):
        semitones = self.NOTE_MAP[note_name] + (octave - 4) * 12 - 9
        return self.REFERENCE_A4 * (2 ** (semitones / 12.0))

    def build_scale(self, root, mode_name, octave):
        intervals = self.MODES.get(mode_name, self.MODES['aeolian'])
        freqs = []
        unique_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        root_idx = unique_notes.index(root)
        for i in intervals:
            note_idx = (root_idx + i) % 12
            octave_bump = (root_idx + i) // 12
            freqs.append(self.get_note_freq(unique_notes[note_idx], octave + octave_bump))
        return freqs

    def get_chord_progression(self, progression_type):
        progs = {
            'trance_uplifting': [('A', 'minor'), ('F', 'major'), ('C', 'major'), ('G', 'major')],
            'techno_dark':      [('C', 'minor'), ('C', 'minor'), ('Eb', 'major'), ('Ab', 'major')],
            'acid_line':        [('E', 'minor'), ('E', 'minor'), ('D', 'major'), ('A', 'major')]
        }
        return progs.get(progression_type, progs['trance_uplifting'])

    def get_functional_chord(self, scale_freqs, function='tonic'):
        indices = {
            'tonic': 0, 'supertonic': 1, 'mediant': 2, 'subdominant': 3,
            'dominant': 4, 'submediant': 5, 'leading': 6
        }
        root_idx = indices.get(function, 0)
        chord = [
            scale_freqs[root_idx],
            scale_freqs[(root_idx + 2) % 7],
            scale_freqs[(root_idx + 4) % 7]
        ]
        return chord

    def generate_narrative_path(self, length_bars=8):
        path = []
        for i in range(length_bars):
            progress = i / length_bars
            if progress < 0.25:
                path.append('tonic')
            elif progress < 0.5:
                path.append('subdominant')
            elif progress < 0.75:
                path.append('dominant')
            else:
                path.append('tonic')
        return path

    def generate_motif(self, scale_freqs, length=8, jumps=False):
        motif = []
        current_idx = 0

        for _ in range(length):
            if jumps:
                choices = [0, 2, 4, 6]
                idx = np.random.choice(choices)
            else:
                step = np.random.choice([-1, 0, 1])
                current_idx = (current_idx + step) % len(scale_freqs)
                idx = current_idx

            motif.append(scale_freqs[idx])
        return motif

    def apply_humanization(self, motif, tension_level=0.5):
        modified = []
        for freq in motif:
            if tension_level > 0.8 and np.random.random() > 0.6:
                modified.append(freq * 2)
            else:
                detune = 1.0 + (np.random.uniform(-0.002, 0.002) * tension_level)
                modified.append(freq * detune)
        return modified

    def get_chord_notes(self, root, mode, octave, chord_type='triad'):
        scale = self.build_scale(root, mode, octave)
        indices = [0, 2, 4]
        if chord_type == 'seventh':
            indices.append(6)

        chord = [scale[i % len(scale)] for i in indices]
        if np.random.random() > 0.5:
            chord[0] *= 2
        return chord

    @property
    def EMOTION_PRESETS(self):
        return {
            "cyberpunk": {
                "root": "F#", "mode": "phrygian", "octave": 2,
                "progression": [0, 1, 0, 1],
                "energy": 0.9
            },
            "melancholy": {
                "root": "A", "mode": "aeolian", "octave": 3,
                "progression": [0, 5, 3, 4],
                "energy": 0.3
            },
            "heroic": {
                "root": "C", "mode": "lydian", "octave": 4,
                "progression": [0, 3, 4, 0],
                "energy": 0.7
            },
            "deep_space": {
                "root": "D", "mode": "locrian", "octave": 2,
                "progression": [0, 4, 3, 4],
                "energy": 0.5
            }
        }

    def get_emotional_setup(self, emotion_name):
        presets = self.EMOTION_PRESETS
        config = presets.get(emotion_name, presets["cyberpunk"])

        scale_freqs = self.build_scale(config["root"], config["mode"], config["octave"])

        progression_freqs = []
        unique_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        root_idx = unique_notes.index(config["root"])

        for interval in config["progression"]:
            chord_root_idx = (root_idx + self.MODES[config["mode"]][interval]) % 12
            chord_root = unique_notes[chord_root_idx]
            chord = self.get_chord_notes(chord_root, config["mode"], config["octave"])
            progression_freqs.append(chord)

        return scale_freqs, progression_freqs

    @staticmethod
    def transpose(freq_list, semitones):
        factor = 2 ** (semitones / 12.0)
        return [f * factor for f in freq_list]

    @staticmethod
    def frequency_to_midi(freq):
        return int(69 + 12 * np.log2(freq / 440.0))
