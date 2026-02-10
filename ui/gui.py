# music_maker/ui/gui.py
import tkinter as tk
from tkinter import ttk
import threading
import os
import sys

# Asegurar import desde paquete
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from core.main import Renderer

def run_gui(renderer=None):
    """
    Arranca una GUI mínima. Si no se pasa renderer, crea uno nuevo.
    """
    if renderer is None:
        renderer = Renderer()

    root = tk.Tk()
    root.title("Music Maker - GUI")

    frm = ttk.Frame(root, padding=12)
    frm.grid()

    ttk.Label(frm, text="Style").grid(column=0, row=0, sticky="w")
    style_var = tk.StringVar(value="techno")
    ttk.Entry(frm, textvariable=style_var).grid(column=1, row=0)

    ttk.Label(frm, text="Emotion").grid(column=0, row=1, sticky="w")
    emotion_var = tk.StringVar(value="cyberpunk")
    ttk.Entry(frm, textvariable=emotion_var).grid(column=1, row=1)

    ttk.Label(frm, text="Duration s").grid(column=0, row=2, sticky="w")
    dur_var = tk.IntVar(value=30)
    ttk.Entry(frm, textvariable=dur_var).grid(column=1, row=2)

    status = tk.StringVar(value="Listo")
    ttk.Label(frm, textvariable=status).grid(column=0, row=4, columnspan=2, pady=(8,0))

    def render_thread():
        try:
            status.set("Renderizando...")
            out, _ = renderer.render(duration_sec=dur_var.get(), style=style_var.get(), emotion=emotion_var.get())
            status.set(f"Render completado: {os.path.basename(out)}")
        except Exception:
            status.set("Error durante render. Revisa logs.")

    def on_render():
        t = threading.Thread(target=render_thread, daemon=True)
        t.start()

    ttk.Button(frm, text="Render", command=on_render).grid(column=0, row=3, columnspan=2, pady=(6,0))

    root.mainloop()

if __name__ == "__main__":
    # Permite ejecutar el archivo directamente: python .\music_maker\ui\gui.py
    run_gui()
