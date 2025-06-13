# teclado_ocular.py
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from collections import deque

# Cargar modelo y datos
model = load_model("modelo_gru.h5")
char_to_idx, idx_to_char, max_seq_len, palabras = joblib.load("autocompletar_datos.pkl")

teclas = [
    list("qwertyuiop"),
    list("asdfghjklñ"),
    list("zxcvbnm ,.")
]

texto = ""
palabra_actual = ""
TIEMPO_SELECCION = 1.5
ultima_fila, ultima_col = -1, -1
ultimo_tiempo = time.time()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
IRIS_DERECHO_ID = [474, 475, 476, 477]

OJO_DERECHO = [33, 160, 158, 133, 153, 144]
OJO_IZQUIERDO = [362, 385, 387, 263, 373, 380]

UMBRAL_EAR = 0.21
MAX_TIEMPO_ENTRE_PARPADEOS = 0.5
tiempos_parpadeos = deque(maxlen=2)

sugerencia_x1, sugerencia_y1 = 10, 70
sugerencia_x2, sugerencia_y2 = 600, 110

font = ImageFont.truetype("arial.ttf", 32)

def autocompletar(palabra_actual):
    prefijo = palabra_actual.lower()
    if not prefijo:
        return ""
    sugerencia = prefijo
    for _ in range(10):
        seq = [char_to_idx.get(c, 0) for c in sugerencia]
        seq = pad_sequences([seq], maxlen=max_seq_len, padding='pre')
        pred = model.predict(seq, verbose=0)
        next_char_idx = np.argmax(pred)
        if next_char_idx == 0:
            break
        next_char = idx_to_char[next_char_idx]
        sugerencia += next_char
        if sugerencia in palabras:
            return sugerencia
    return ""

def detectar_mirada_por_iris(landmarks, img_w, img_h):
    iris_coords = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in IRIS_DERECHO_ID]
    if iris_coords:
        iris_center = np.mean(iris_coords, axis=0).astype(int)
        return tuple(iris_center)
    return None

def calcular_ear(puntos, landmarks, img_w, img_h):
    puntos = [(int(landmarks[p].x * img_w), int(landmarks[p].y * img_h)) for p in puntos]
    A = np.linalg.norm(np.array(puntos[1]) - np.array(puntos[5]))
    B = np.linalg.norm(np.array(puntos[2]) - np.array(puntos[4]))
    C = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def detectar_tecla_seleccionada(img, x, y):
    h, w = img.shape[:2]
    tecla_w = w // 10
    tecla_h = 60
    y_inicio = h - tecla_h * 3 - 20
    if y < y_inicio:
        return -1, -1
    fila = (y - y_inicio) // tecla_h
    col = x // tecla_w
    if fila < 0 or fila > 2 or col < 0 or col > 9:
        return -1, -1
    return int(fila), int(col)

def dibujar_teclado(img, fila_sel, col_sel):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    h, w = img.shape[:2]
    tecla_w = w // 10
    tecla_h = 60
    y_inicio = h - tecla_h * 3 - 20
    for i, fila in enumerate(teclas):
        for j, letra in enumerate(fila):
            x = j * tecla_w
            y = y_inicio + i * tecla_h
            color = (50, 30, 30) if (i != fila_sel or j != col_sel) else (0, 255, 0)
            draw.rectangle([x, y, x + tecla_w - 2, y + tecla_h - 2], fill=color)
            bbox = draw.textbbox((x, y), letra, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((x + (tecla_w - text_w) // 2, y + (tecla_h - text_h) // 2), letra, font=font, fill=(255, 255, 255))
    return np.array(pil_img)

def mostrar_texto_con_pil(img, texto, sugerencia):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle((sugerencia_x1, sugerencia_y1 - 30, sugerencia_x2, sugerencia_y2), outline=(50, 50, 50), width=2)
    draw.text((sugerencia_x1 + 10, sugerencia_y1), f"Sugerencia: {sugerencia}", font=font, fill=(180, 180, 180))
    draw.text((10, 10), f"Texto: {texto}", font=font, fill=(255, 255, 255))
    return np.array(pil_img)

def main():
    global texto, palabra_actual, ultima_fila, ultima_col, ultimo_tiempo
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return

    tiempo_ultima_sugerencia = 0
    sugerencia = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        fila_sel, col_sel = -1, -1
        iris_pos = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                iris_pos = detectar_mirada_por_iris(face_landmarks.landmark, w, h)

                # Detección EAR y doble parpadeo
                ear_izq = calcular_ear(OJO_IZQUIERDO, face_landmarks.landmark, w, h)
                ear_der = calcular_ear(OJO_DERECHO, face_landmarks.landmark, w, h)
                ear = (ear_izq + ear_der) / 2.0

                if iris_pos:
                    cv2.circle(frame, iris_pos, 5, (255, 0, 255), -1)
                    fila_sel, col_sel = detectar_tecla_seleccionada(frame, iris_pos[0], iris_pos[1])

                    x_iris, y_iris = iris_pos
                    en_sugerencia = (sugerencia_x1 <= x_iris <= sugerencia_x2) and (sugerencia_y1 - 30 <= y_iris <= sugerencia_y2)

                    if ear < UMBRAL_EAR:
                        tiempos_parpadeos.append(time.time())
                        if len(tiempos_parpadeos) == 2 and en_sugerencia:
                            intervalo = tiempos_parpadeos[1] - tiempos_parpadeos[0]
                            if intervalo < MAX_TIEMPO_ENTRE_PARPADEOS:
                                texto += sugerencia[len(palabra_actual):] + " "
                                palabra_actual = ""
                                tiempos_parpadeos.clear()

        if fila_sel != -1 and col_sel != -1:
            if (fila_sel, col_sel) == (ultima_fila, ultima_col):
                if time.time() - ultimo_tiempo >= TIEMPO_SELECCION:
                    letra = teclas[fila_sel][col_sel]
                    if letra in [",", ".", " "]:
                        texto += letra
                        palabra_actual = ""
                    else:
                        texto += letra
                        palabra_actual += letra
                    ultimo_tiempo = time.time() + 1
            else:
                ultima_fila, ultima_col = fila_sel, col_sel
                ultimo_tiempo = time.time()

        # Autocompletar solo con palabra_actual
        if time.time() - tiempo_ultima_sugerencia > 0.5:
            sugerencia = autocompletar(palabra_actual)
            tiempo_ultima_sugerencia = time.time()

        frame = dibujar_teclado(frame, fila_sel, col_sel)
        frame = mostrar_texto_con_pil(frame, texto, sugerencia)

        cv2.imshow("Teclado Virtual con Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
