# 🤖 Neural Music Architect Pro

**Neural Music Architect Pro** es un motor de composición algorítmica de vanguardia diseñado para generar pistas de música electrónica de alta fidelidad utilizando técnicas de síntesis de audio digital (DSP) y redes neuronales básicas para la toma de decisiones estructuradas.

El sistema es capaz de crear composiciones completas por actos (Intro, Build-up, Drop, Outro) con un estilo emocional definido.

## Estructura del proyecto

Listado de rutas de carpetas para el volumen Windows
El número de serie del volumen es 788B-24D1
/MUSIC_MAKER/
│   .gitignore
│   README.md
│   requirements.txt
│   __init__.py
│   
├───compositor
│   │   arranger.py
│   │   scales.py
│   │   sequencer.py
│   │   __init__.py
│   │
│   └───__pycache__
│           arranger.cpython-313.pyc
│           scales.cpython-313.pyc
│           sequencer.cpython-313.pyc
│           __init__.cpython-313.pyc
│
├───core
│   │   logger.py
│   │   main.py
│   │   __init__.py
│   │
│   └
│           
│           
│
├───engine
│   │   effects.py
│   │   filters.py
│   │   oscillators.py
│   │   percussion.py
│   │   __init__.py
│   │
│   └
│           
│           
│           
│           
│       
│
├───exports
│       
│
└───logs


---

## 🛠️ Tecnologías Utilizadas

* **Python 3.x**: Lenguaje base.
* **NumPy**: Procesamiento numérico y manipulación de señales de audio a alta velocidad.
* **SciPy**: Operaciones avanzadas de DSP (filtros IIR, escritura de archivos WAV).
* **Tkinter**: Interfaz gráfica de usuario (GUI) para control en tiempo real.

---

## 🎹 Módulos del Motor

| Módulo | Descripción |
| --- | --- |
| `Arranger` | Gestiona la narrativa de la canción, estructura por actos y tensión emocional. |
| `Oscillators` | Motor de síntesis: Supersaw, Acid 303, Pads ambientales y Síntesis Vocal. |
| `Percussion` | Generador de ritmos: Kicks de modelado físico, Snares, Hats y patrones IDM/Euclidianos. |
| `Filters` | Modelado de filtros analógicos: Moog Ladder, Auto-Wah, Formantes para voz. |
| `Effects` | Cadena de Mastering: Compresor Sidechain, Reverb y Limitador final. |

---

## 🚀 Instalación y Uso

### Prerrequisitos

Necesitas tener instalado Python. Recomendamos usar un entorno virtual.

### Pasos

1. **Clonar el repositorio**:
```bash
git clone https://github.com/tu-usuario/neural-music-architect.git
cd neural-music-architect

```


2. **Instalar dependencias**:
```bash
pip install numpy scipy

```


3. **Ejecutar la aplicación**:
```bash
python core/main.py

```



---

## 🖥️ Interfaz Gráfica (GUI)

La interfaz permite controlar en tiempo real los siguientes parámetros:

* **Estilo musical**: Techno, Trance, Cyberpunk, Jazz.
* **Duración de la pista**: entre 1 y 10 minutos, la estructura se adapta segun la longitud.
* **Nivel de saturación (Warmth)**.
* **Amplitud estéreo (Haas Effect)**.

---

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más información.