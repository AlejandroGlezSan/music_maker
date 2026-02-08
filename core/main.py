import sys
import tkinter as tk
from tkinter import ttk, messagebox
import os
import numpy as np
from scipy.io import wavfile
from compositor.arranger import Arranger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MusicMakerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Engine - Pro Station")
        self.root.geometry("400x450")
        
        self.arranger = Arranger()
        self._setup_ui()

    def _setup_ui(self):
        # Selección de Estilo
        tk.Label(self.root, text="Estilo Musical:", font=('Arial', 10, 'bold')).pack(pady=5)
        self.style_var = tk.StringVar(value="techno")
        styles = ["techno", "trance", "dubstep", "brostep"]
        self.style_combo = ttk.Combobox(self.root, textvariable=self.style_var, values=styles, state="readonly")
        self.style_combo.pack(pady=5)

        # Duración
        tk.Label(self.root, text="Duración (segundos):").pack(pady=5)
        self.duration_scale = tk.Scale(self.root, from_=10, to=120, orient=tk.HORIZONTAL)
        self.duration_scale.set(30)
        self.duration_scale.pack(pady=5)

        # Botón de Generación
        self.btn_generate = tk.Button(self.root, text="GENERAR TRACK", command=self.generate_music, 
                                     bg="#2ecc71", fg="white", font=('Arial', 12, 'bold'), height=2)
        self.btn_generate.pack(pady=30, fill=tk.X, padx=50)

        self.status_label = tk.Label(self.root, text="Listo para crear", fg="gray")
        self.status_label.pack(pady=10)

    def generate_music(self):
        try:
            style = self.style_var.get()
            duration = self.duration_scale.get()
            
            self.status_label.config(text=f"Generando {style}...", fg="blue")
            self.root.update()

            # Llamada al motor de composición
            audio_data = self.arranger.compose(style, duration)
            
            # Normalización final por seguridad
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
            
            # Crear carpeta de salida si no existe
            output_dir = "exports"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            filename = f"{output_dir}/{style}_track_{np.random.randint(1000, 9999)}.wav"
            wavfile.write(filename, 44100, (audio_data * 32767).astype(np.int16))

            messagebox.showinfo("Éxito", f"Track generado en:\n{filename}")
            self.status_label.config(text="Track guardado correctamente", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error Crítico", f"Ha ocurrido un error:\n{str(e)}")
            self.status_label.config(text="Error en la generación", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicMakerGUI(root)
    root.mainloop()