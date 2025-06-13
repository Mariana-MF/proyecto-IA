# proyecto-IA
Prototipo de interfaz visual que se utiliza por medio del seguimiento ocular (eye tracking) usando técnicas de IA.

# crea un entorno virtual e instala dependencias:
python -m venv venv
venv\Scripts\activate  (en Windows)
pip install opencv-python mediapipe numpy pillow keras joblib

# entrena el modelo
entrenar_modelo.py

# ejecuta el archivo principal
python teclado_ocular.py

# función
MediaPipe detecta puntos del rostro y la posición del iris.
Se calcula el EAR (Eye Aspect Ratio) para detectar parpadeos.
Al mirar una letra durante 1.5 segundos, se selecciona.
El sistema analiza la palabra actual y la envía como prefijo al modelo GRU.
Si el usuario realiza un doble parpadeo sobre la sugerencia, la palabra se completa automáticamente.
