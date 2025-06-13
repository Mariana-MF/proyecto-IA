# entrenar_modelo.py
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import joblib

palabras = [
    "adios", "amor", "amigo", "auto", "avanzar", "avion", "azul",
    "bajo", "banco", "barco", "barrio", "bebe", "bien", "bueno",
    "casa", "cielo", "coche", "color", "comer", "comida", "corazon", "correr",
    "decir", "dedo", "deporte", "dia", "dinero", "dormir", "dulce",
    "escribir", "escuela", "espejo", "espacio", "estrella", "estudio",
    "facil", "familia", "feliz", "fiesta", "fuego", "futuro",
    "gato", "gente", "globo", "grande", "gritar", "grupo",
    "hablamos", "habito", "habitacion", "hablar", "hacer", "habil",
    "hambre", "hamaca", "helado", "hermano", "hermoso", "heroina",
    "hola", "holanda", "hotel", "hoy", "huevo", "humano",
    "idea", "iglesia", "invierno", "isla",
    "joven", "juego", "jugar",
    "lago", "libro", "lento", "luz",
    "madre", "mano", "mar", "mañana",
    "niño", "noche", "nuevo", "numero",
    "ojo", "oro", "oso",
    "padre", "palabra", "paz", "perro", "persona", "plaza", "playa",
    "que", "querer", "quizas",
    "raton", "rey", "ropa",
    "saber", "salud", "sol", "sueño",
    "tarde", "taza", "tiempo", "tierra",
    "universo", "usar",
    "verde", "vida", "viento", "vino",
    "voz", "vuelo",
    "zapato", "zanahoria", "zarpar"
]

caracteres = sorted(list(set("".join(palabras))))
char_to_idx = {c: i+1 for i, c in enumerate(caracteres)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

sequences, targets = [], []
for palabra in palabras:
    for i in range(1, len(palabra)):
        seq = [char_to_idx[c] for c in palabra[:i]]
        target = char_to_idx[palabra[i]]
        sequences.append(seq)
        targets.append(target)

max_seq_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')
y = to_categorical(targets, num_classes=len(char_to_idx)+1)

model = Sequential([
    Embedding(input_dim=len(char_to_idx)+1, output_dim=16, input_length=max_seq_len),
    GRU(32),
    Dense(len(char_to_idx)+1, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, y, epochs=20, verbose=1)

model.save("modelo_gru.h5")
joblib.dump((char_to_idx, idx_to_char, max_seq_len, palabras), "autocompletar_datos.pkl")