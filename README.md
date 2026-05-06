# **🍷 CLASIFICADOR DE CALIDAD DE VINO**
Este proyecto permite predecir la calidad de vino al basarse en sus propiedades químicas. Para dicha predicción se utilizan modelos de clasificación como Regresión Logística y Suport Vector Machine (SVM).
El flujo del proyecto va desde un análisis exploratorio (EDA), entrenamiento de modelo y predicción de los resultados.

## 1. **Estructura del Proyecto:**
* `data/`: Contiene el dataset original `WineQT.csv`.
* `models/`: Contiene el modelo entrenado (`modelo.joblib`) y el escalador de datos (`scaler.joblib`).
* `src/`: Carpeta con el código fuente en módulos:
    * `eda.py`: Limpieza de datos y visualizaciones.
    * `entrenamiento.py`: Procesamiento de datos entrenamiento de modelos y selección del mejor.
    * `prueba.py`: Carga del modelo guardado, métricas finales y matriz de confusión.

## 2. **Requisitos de Instalación:**
Para poder ejecutar este proyecto, se necesita tener instalado Python y pip install.

## 3. **Ejecución del Proyecto:**
Primero, se debe clonar el repositorio. Una vez la dirección del directorio sea '/WINE_REPO_SC', continuar:

- Se deben instalar las librerias necesarias con el siguiente comando:
```pip install -r requirements.txt```

- Posteriormente se puede seguir con el flujo del proyecto:

### Análisis de datos:
```python src/eda.py```

### Entrenamiento y Selección de Modelo:
```python src/entrenamiento.py```

### Evaluación final:
```python src/prueba.py```

## 4. **Resultados esperados:**
El modelo que fue seleccionado como ganador fue el de Support Vector Machine (SVM) debido a su gran capacidad para manejar relaciones no lineales en los datos químicos, obteniendo un Accuracy superior a comparación de la Regresión Logística

**Desarrollado por:** Samantha Castro