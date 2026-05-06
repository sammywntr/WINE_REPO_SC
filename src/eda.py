"""
PREPROCESAMIENTO DE LOS DATOS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cargar_datos(ruta):
    df = pd.read_csv(ruta)

    if "Id" in df.columns:
        df = df.drop(columns = ["Id"])

    print(f"\n DATASET CARGADO EXITOSAMENTE. \n Pre-visualización del dataset: \n {df.head(5)}")
    return df


#La limpieza de datos no es necesaria ya que no hay valores nulos, duplicados y todos los registros estan en su formato correcto.


def estadisticas_descriptivas(df):
    print(f"\n ESTADISTICAS DESCRIPTIVAS: \n {df.describe()}")


def visualizacion_distribuciones(df):
    sns.set_theme(style = "whitegrid")

    #Histogramas
    df.hist(figsize = (13, 6), bins = 20)
    plt.suptitle("Histograma de Variables")
    plt.tight_layout()
    plt.show()

    #Boxplots
    plt.figure(figsize = (13, 6))
    sns.boxplot(data = df, orient = "h")
    plt.suptitle("Boxplots de Variables")
    plt.show()

    #Matriz de correlación
    plt.figure(figsize = (13, 7))
    sns.heatmap(df.corr(), annot = True, cmap = "coolwarm", fmt = ".2f")
    plt.suptitle("Matriz de Correlación")
    plt.show()


if __name__ == "__main__":
    ruta_csv = "data/WineQT.csv"

    datos = cargar_datos(ruta_csv)
    estadisticas_descriptivas(datos)
    visualizacion_distribuciones(datos)