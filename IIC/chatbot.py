import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar el archivo CSV
df = pd.read_csv('hist.csv')

# Vectorización de las preguntas
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Pregunta'])

# Función para obtener la respuesta más similar usando similitud de coseno
def get_answer(user_question):
    # Transformar la pregunta del usuario en el mismo espacio de características
    user_vector = vectorizer.transform([user_question])
    
    # Calcular la similitud de coseno entre la pregunta del usuario y las preguntas en el CSV
    similarities = cosine_similarity(user_vector, X)
    
    # Encontrar el índice de la pregunta más similar
    most_similar_index = np.argmax(similarities)
    
    # Obtener la respuesta correspondiente
    return df.iloc[most_similar_index]['Respuesta']

# Interacción con el usuario
while True:
    user_input = input("Tú (Para terminar escribe salir): ")
    if user_input.lower() == 'salir':
        print("Chatbot: ¡Adiós!")
        break
    response = get_answer(user_input)
    print(f"Chatbot: {response}")
