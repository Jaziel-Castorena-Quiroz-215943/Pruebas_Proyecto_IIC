import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Crear datos simulados
x = np.linspace(0, 50, 500)
y = np.sin(x)

# Crear secuencias
time_steps = 10
data_x, data_y = [], []
for i in range(len(y) - time_steps):
    data_x.append(y[i:i + time_steps])
    data_y.append(y[i + time_steps])

data_x = np.array(data_x)
data_y = np.array(data_y)

# Dividir en conjuntos de entrenamiento y prueba
split = int(len(data_x) * 0.8)
x_train, x_test = data_x[:split], data_x[split:]
y_train, y_test = data_y[:split], data_y[split:]

# Redimensionar para LSTM
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Crear modelo LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, 1)),
    Dense(1)
])

# Compilar y entrenar
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=20, batch_size=16)

# Predecir y graficar resultados
predicted = model.predict(x_test)
plt.plot(range(len(y)), y, label='Original Data')
plt.plot(range(split, len(y)), predicted, label='Predictions', linestyle='dashed')
plt.legend()
plt.show()
