# Frecuencias de notas base para construir acordes
CHORDS = {
    # Menores
    'Am': [220.00, 261.63, 329.63], # La menor
    'Dm': [146.83, 174.61, 220.00], # Re menor
    'Em': [164.81, 196.00, 246.94], # Mi menor
    'Fm': [174.61, 207.65, 261.63], # Fa menor
    'Cm': [130.81, 155.56, 196.00], # Do menor (El que faltaba)
    'Bm': [123.47, 146.83, 185.00], # Si menor
    
    # Mayores
    'C':  [130.81, 164.81, 196.00], # Do Mayor
    'F':  [174.61, 220.00, 261.63], # Fa Mayor
    'G':  [196.00, 246.94, 293.66], # Sol Mayor
    'Bb': [116.54, 146.83, 174.61], # Si bemol Mayor
    'Ab': [103.83, 130.81, 155.56]  # La bemol Mayor
}

# Diferentes "estados de ánimo" para las canciones
PROGRESSIONS = [
    ["Am", "F", "C", "G"],    # Épica / Himno
    ["Dm", "Bb", "F", "C"],   # Melancólica profunda
    ["Cm", "Ab", "F", "G"],   # Techno Oscuro (Cm corregido)
    ["Am", "Em", "F", "G"],   # Ascendente / Trance
    ["Fm", "Cm", "Ab", "Bb"]  # Dramática
]