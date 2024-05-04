import numpy as np
#from sklearn.datasets import load_files
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Cargar conjunto de datos IMDb
movie_reviews_data = pd.read_csv('Datathon 2024 - Reto Hey - Dataset Público - Sheet1.csv')

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
   movie_reviews_data.data, movie_reviews_data.target, test_size=0.2, random_state=42)


# Vectorización de texto usando CountVectorizer
vectorizer = CountVectorizer(max_features=10000)  # Limitar a las 10000 características más frecuentes
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# Entrenamiento del modelo SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vectorized, y_train)


# Predicción de emociones en el conjunto de prueba
y_pred = svm_model.predict(X_test_vectorized)


# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo SVM:", accuracy)


# Ejemplo de predicción de sentimientos en un texto de entrada
texto_ejemplo = ["Esta película fue increíble, me encantó cada momento."]
texto_vectorizado = vectorizer.transform(texto_ejemplo)
