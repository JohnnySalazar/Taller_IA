import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import pandas as pd
import joblib

# Cargar el modelo desde el archivo
model_filename = 'modelo_entrenado.pkl'
loaded_model = joblib.load(model_filename)

# Cargar nuevos datos desde un archivo Excel
excel_file = 'nuevos_datos.xlsx'
new_data = pd.read_excel(excel_file)

# Realizar predicciones en los nuevos datos
predictions = loaded_model.predict(new_data)
probabilities = loaded_model.predict_proba(new_data)  # Si el modelo es de clasificación

# Imprimir las predicciones y argumentos
for pred, prob in zip(predictions, probabilities):
    print("Predicción:", pred)
    print("Probabilidades:", prob)
    print()

