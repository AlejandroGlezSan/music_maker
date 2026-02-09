import sys
import os
import numpy as np
import threading
from scipy.io import wavfile
import tkinter as tk
from tkinter import ttk, messagebox
import time

# --- 1. MOTOR DE RUTAS INTELIGENTE (FIX) ---
# Detectamos la carpeta raíz del proyecto independientemente de dónde se ejecute
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PATH)

def check_system_integrity():
    required_paths = [
        'compositor/arranger.py',
        'compositor/scales.py',
        'engine/oscillators.py',
        'engine/effects.py'
    ]
    missing = []
    for path in required_paths:
        full_path = os.path.join(BASE_PATH, path)
        if not os.path.exists(full_path):
            missing.append(path)
    return missing

# Intentamos importar ahora que la ruta está corregida
try:
    from compositor.arranger import Arranger
    from engine.effects import Effects as Ef
except ModuleNotFoundError as e:
    print(f"\n[!] ERROR DE ESTRUCTURA: No se pudo cargar el módulo. {e}")
    print(f"Ruta base detectada: {BASE_PATH}")
    sys.exit()

# --- 3. CLASE PRINCIPAL (ACTUALIZADA) ---

class NeuralStudioPro:
    def __init__(self, root):
        self.root = root
        self.root.title("NEURAL MUSIC ARCHITECT - v12.1 STABLE")
        self.root.geometry("700x900")
        self.root.configure(bg="#050507")
        
        # Verificar integridad antes de empezar
        missing = check_system_integrity()
        if missing:
            error_msg = "Archivos faltantes:\n" + "\n".join(missing)
            messagebox.showerror("INTEGRITY ERROR", error_msg)
            # No cerramos, para que la consola muestre el error
        
        self.arranger = Arranger()
        self._init_styles()
        self._build_ui()

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from compositor.arranger import Arranger
from engine.effects import Effects as Ef

class NeuralStudioPro:
    def __init__(self, root):
        self.root = root
        self.root.title("NEURAL MUSIC ARCHITECT - v12.0 STORYTELLER")
        self.root.geometry("700x900")
        self.root.configure(bg="#050507")
        
        self.arranger = Arranger()
        self._init_styles()
        self._build_ui()

    def _init_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#050507")
        style.configure("TLabel", background="#050507", foreground="#00ffcc", font=("Segoe UI", 10))
        style.configure("Horizontal.TScale", background="#050507")

    def _build_ui(self):
        # --- HEADER ---
        header = tk.Frame(self.root, bg="#050507", pady=20)
        header.pack(fill="x")
        tk.Label(header, text="NEURAL ARCHITECT", font=("Orbitron", 24, "bold"), 
                 bg="#050507", fg="#00ffcc").pack()
        tk.Label(header, text="ALGORITHMIC STORYTELLING ENGINE", font=("Consolas", 9), 
                 bg="#050507", fg="#0088aa").pack()

        # --- PANEL PRINCIPAL ---
        main_panel = tk.Frame(self.root, bg="#050507", padx=30)
        main_panel.pack(fill="both", expand=True)

        # 1. Configuración de la Historia
        story_frame = tk.LabelFrame(main_panel, text=" PARÁMETROS NARRATIVOS ", bg="#050507", 
                                   fg="#00ffcc", font=("Segoe UI", 10, "bold"), padx=15, pady=15)
        story_frame.pack(fill="x", pady=10)

        tk.Label(story_frame, text="ESTILO MUSICAL:").grid(row=0, column=0, sticky="w")
        self.style_var = tk.StringVar(value="techno")
        style_box = ttk.Combobox(story_frame, textvariable=self.style_var, 
                                values=["techno", "trance", "dubstep", "idm"])
        style_box.grid(row=0, column=1, sticky="ew", pady=5)

        tk.Label(story_frame, text="EMOCIÓN BASE:").grid(row=1, column=0, sticky="w")
        self.emotion_var = tk.StringVar(value="cyberpunk")
        emotion_box = ttk.Combobox(story_frame, textvariable=self.emotion_var, 
                                  values=["cyberpunk", "melancholy", "heroic", "deep_space"])
        emotion_box.grid(row=1, column=1, sticky="ew", pady=5)

        tk.Label(story_frame, text="DURACIÓN (SEG):").grid(row=2, column=0, sticky="w")
        self.dur_slider = tk.Scale(story_frame, from_=15, to=120, orient="horizontal", 
                                  bg="#050507", fg="#00ffcc", highlightthickness=0)
        self.dur_slider.set(30)
        self.dur_slider.grid(row=2, column=1, sticky="ew")

        # 2. Masterización y Color
        fx_frame = tk.LabelFrame(main_panel, text=" MASTERING & TEXTURE ", bg="#050507", 
                                fg="#00ffcc", font=("Segoe UI", 10, "bold"), padx=15, pady=15)
        fx_frame.pack(fill="x", pady=10)

        tk.Label(fx_frame, text="CALOR ANALÓGICO:").grid(row=0, column=0, sticky="w")
        self.warmth_slider = tk.Scale(fx_frame, from_=0, to=100, orient="horizontal", bg="#050507", fg="#00ffcc")
        self.warmth_slider.set(30)
        self.warmth_slider.grid(row=0, column=1, sticky="ew")

        tk.Label(fx_frame, text="ANCHO ESTÉREO:").grid(row=1, column=0, sticky="w")
        self.stereo_slider = tk.Scale(fx_frame, from_=0, to=100, orient="horizontal", bg="#050507", fg="#00ffcc")
        self.stereo_slider.set(50)
        self.stereo_slider.grid(row=1, column=1, sticky="ew")

        # 3. Consola de Proceso
        self.console = tk.Text(main_panel, height=12, bg="#000", fg="#00ff88", 
                              font=("Consolas", 9), borderwidth=0, padx=10, pady=10)
        self.console.pack(fill="x", pady=10)
        self.log("SISTEMA LISTO. ESPERANDO SECUENCIA...")

        # --- BOTÓN DE RENDER ---
        self.render_btn = tk.Button(self.root, text="GENERATE UNIVERSE", font=("Orbitron", 14, "bold"),
                                   bg="#00ffcc", fg="#000", activebackground="#00ccaa",
                                   command=self.start_render_thread, cursor="hand2")
        self.render_btn.pack(fill="x", side="bottom", padx=30, pady=30)

    def log(self, message):
        self.console.insert(tk.END, f"> {message}\n")
        self.console.see(tk.END)
        self.root.update_idletasks()

    def start_render_thread(self):
        self.render_btn.config(state="disabled", text="PROCESSING...")
        thread = threading.Thread(target=self.render_audio)
        thread.start()

    def render_audio(self):
        try:
            style = self.style_var.get()
            emotion = self.emotion_var.get()
            duration = self.dur_slider.get()

            # 1. Composición Narrativa
            self.log(f"INICIANDO COMPOSICIÓN: {style.upper()} / {emotion.upper()}")
            audio = self.arranger.compose(style, duration)
            
            # 2. Aplicar Mastering personalizado desde la UI
            self.log("APLICANDO CADENA DE MASTERING...")
            
            # Saturación de válvulas
            warmth = self.warmth_slider.get() / 100.0
            audio = Ef.tube_warmth(audio, drive=1.0 + (warmth * 2), warmth=warmth)
            
            # Estéreo Haas
            width = 1.0 + (self.stereo_slider.get() / 100.0)
            audio = Ef.stereo_widener(audio, width=width)
            
            # Limitador de seguridad
            audio = Ef.master_limiter(audio)

            # 3. Exportación
            if not os.path.exists("exports"): os.makedirs("exports")
            timestamp = int(time.time())
            filename = f"exports/neural_track_{timestamp}.wav"
            
            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(filename, 44100, audio_int16)

            self.log(f"RENDERIZADO EXITOSO: {filename}")
            messagebox.showinfo("SUCCESS", f"Canción generada correctamente:\n{filename}")

        except Exception as e:
            self.log(f"ERROR CRÍTICO: {str(e)}")
            messagebox.showerror("ENGINE FAILURE", f"Error en el motor: {str(e)}")
        
        finally:
            self.render_btn.config(state="normal", text="GENERATE UNIVERSE")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralStudioPro(root)
    root.mainloop()