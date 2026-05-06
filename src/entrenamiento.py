"""
ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from eda import cargar_datos


def preparacion_datos(df, target_col): #-> Incluye escalado de datos para el modelo SVM
    x = df.drop(columns = [target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

    escalado = StandardScaler()
    x_train_escalado = escalado.fit_transform(x_train)
    x_test_escalado = escalado.transform(x_test)

    #Almacenamiento de escalado para poder volver a utilizarlo.
    joblib.dump(escalado, "models/scaler.joblib")
    print(f"\nEL ESCALADOR SE ALMACENÓ EXITOSAMENTE EN: 'models/scaler.joblib'")

    return x_train_escalado, x_test_escalado, y_train, y_test


def calcular_metricas(y_real, y_pred, modelo):
    print(f"\nRESULTADOS DE MÉTRICAS DEL MODELO: {modelo}")

    accuracy = accuracy_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred, average = "weighted", zero_division = 0)
    recall = recall_score(y_real, y_pred, average = "weighted", zero_division = 0)
    f1 = f1_score(y_real, y_pred, average = "weighted", zero_division = 0)

    print(f"- Accuracy Score: {accuracy: .4f}")
    print(f"- Precision Score: {precision: .4f}")
    print(f"- Recall Score: {recall: .4f}")
    print(f"- F1 Score: {f1: .4f}")

    return accuracy


def entrenamiento_modelos(x_train, y_train, x_test, y_test):
    log_reg = LogisticRegression(max_iter = 1000, random_state = 42)
    svm = SVC(kernel = "rbf", random_state = 42)

    print("\nEntrenando modelo de Regresión Logistica...")
    log_reg.fit(x_train, y_train)
    print("Entrenando modelo de Support Vector Machine (SVM)...")
    svm.fit(x_train, y_train)

    y_pred_lr = log_reg.predict(x_test) #-> Se realiza una primer predicción para poder seleccionar el modelo
    y_pred_svm = svm.predict(x_test)

    #Selección del mejor modelo
    acc_lr = calcular_metricas(y_test, y_pred_lr, "Regresión Logística")
    acc_svm = calcular_metricas(y_test, y_pred_svm, "SVM")

    if acc_lr > acc_svm:
        print("\n¡El mejor modelo es el de regresión Logística!")
        return log_reg
    else:
        print("\n¡El mejor modelo es el de Support Vector Machine (SVM)!")
        return svm
    

def guardar_modelo(modelo, ruta):
    joblib.dump(modelo, ruta)
    print(f"El modelo fue guardado exitosamente en: {ruta}")


if __name__ == "__main__":
    objetivo = "quality"
    df = cargar_datos("data/WineQT.csv")
    x_train, x_test, y_train, y_test = preparacion_datos(df, target_col = objetivo)

    mejor_modelo = entrenamiento_modelos(x_train, y_train, x_test, y_test)
    guardar_modelo(mejor_modelo, "models/modelo.joblib")