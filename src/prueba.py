"""
PRUEBA DEL MODELO
"""

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from eda import cargar_datos


def cargar_modelo(ruta_modelo, ruta_escalador):
    modelo = joblib.load(ruta_modelo)
    escalador = joblib.load(ruta_escalador)

    print("\n¡El modelo y el escalador fueron importados exitosamente!")
    return modelo, escalador


def evaluacion_prueba(modelo, escalador,x_test):
    x_test_escalado = escalador.transform(x_test)
    y_pred = modelo.predict(x_test_escalado)

    print("\n¡Evaluación completada!")
    return y_pred


def matriz_confusión(y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize = (13, 6))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Purples")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción del modelo")
    plt.ylabel("Datos reales")
    plt.tight_layout()
    plt.show()

#Reportar métricas finales
def metricas_finales(y_real, y_pred):
    print("\nMÉTRICAS FINALES DEL MODELO:")

    accuracy = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred, average = "weighted", zero_division = 0)
    recall = recall_score(y_real, y_pred, average = "weighted", zero_division = 0)
    f1 = f1_score(y_real, y_pred, average = "weighted", zero_division = 0)

    print(f"- Accuracy Score: {accuracy: .4f}")
    print(f"- Precision Score: {precision: .4f}")
    print(f"- Recall Score: {recall: .4f}")
    print(f"- F1 Score: {f1: .4f}")


if __name__ == "__main__":
    objetivo = "quality"
    df = cargar_datos("data/WineQT.csv")
    x = df.drop(columns = [objetivo])
    y = df[objetivo]
    _, x_test, _, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    modelo_ganador, escalador = cargar_modelo("models/modelo.joblib", "models/scaler.joblib")
    prediccion = evaluacion_prueba(modelo_ganador, escalador, x_test)
    metricas_finales(y_test, prediccion)
    matriz_confusión(y_test, prediccion)