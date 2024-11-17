import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Cargar los datos de IMDB
max_features = 10000  # número máximo de palabras a considerar en el vocabulario
maxlen = 200  # cortamos las reseñas después de 200 palabras

# Cargar datos de entrenamiento y prueba
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Rellenar las secuencias para que tengan la misma longitud
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Crear el modelo con una capa RNN simple
model_rnn = Sequential()
model_rnn.add(Embedding(max_features, 32))  # capa de Embedding para convertir palabras en vectores
model_rnn.add(SimpleRNN(32))  # Capa RNN con 32 unidades
model_rnn.add(Dense(1, activation='sigmoid'))  # Capa de salida para clasificación binaria

# Compilar el modelo
model_rnn.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history_rnn = model_rnn.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)


from tensorflow.keras.layers import LSTM

# Crear el modelo con una capa LSTM
model_lstm = Sequential()
model_lstm.add(Embedding(max_features, 32))  # Capa de Embedding para convertir palabras en vectores
model_lstm.add(LSTM(32))  # Capa LSTM con 32 unidades
model_lstm.add(Dense(1, activation='sigmoid'))  # Capa de salida para clasificación binaria

# Compilar el modelo
model_lstm.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history_lstm = model_lstm.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluación del modelo RNN
test_loss_rnn, test_acc_rnn = model_rnn.evaluate(x_test, y_test)
print(f'RNN Test Accuracy: {test_acc_rnn}')

# Evaluación del modelo LSTM
test_loss_lstm, test_acc_lstm = model_lstm.evaluate(x_test, y_test)
print(f'LSTM Test Accuracy: {test_acc_lstm}')
