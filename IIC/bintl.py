import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Parámetros de configuración
max_features = 10000  # Número máximo de palabras en el vocabulario
maxlen = 300  # Longitud máxima de las secuencias
embedding_dim = 64  # Dimensión de los vectores de palabras

# Cargar datos de IMDB
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Preprocesar datos
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Definir el modelo con múltiples capas LSTM y regularización
model = Sequential([
    Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=maxlen),
    LSTM(64, return_sequences=True),  # Primera capa LSTM
    Dropout(0.2),
    LSTM(32),  # Segunda capa LSTM
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Capa de salida
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    x_train, y_train, 
    epochs=5, 
    batch_size=128, 
    validation_split=0.2
)

# Evaluar en el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
