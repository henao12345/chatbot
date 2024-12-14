from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado de regresión logística
with open('modelo_lr.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Cargar el vectorizador TF-IDF
tfidf = joblib.load("vectorizador_tfidf.pkl")

# Diccionario con respuestas
respuestas = {
    'saludo': '¡Hola! Bienvenido a la sección de energías renovables. ¿Cómo estás hoy?',
    'preguntar_nombre': 'Soy un asistente virtual especializado en energías renovables. ¿Cómo te llamas tú?',
    # Añadir más etiquetas con sus respectivas respuestas
}

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener el mensaje del formulario
        mensaje = request.form['mensaje']
        
        # Transformar el mensaje usando el vectorizador TF-IDF
        mensaje_tfidf = tfidf.transform([mensaje])
        
        # Realizar la predicción
        prediccion = model.predict(mensaje_tfidf)
        etiqueta_predicha = prediccion[0]

        # Obtener la respuesta asociada a la etiqueta predicha
        respuesta = respuestas.get(etiqueta_predicha, "Lo siento, no entiendo esa pregunta.")  # Predicción no reconocida

        # Mostrar el resultado
        return render_template('index.html', mensaje=mensaje, prediccion=etiqueta_predicha, respuesta=respuesta)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
