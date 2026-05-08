# **🍷 WINE QUALITY CLASSIFIER**
This project allows to predict wine quality based on its chemical properties. Classification models such as Logistic Regression and Support Vector Machine (SVM) are used for this prediction.
The project flow goes from an exploratory data analysis (EDA), model training, and prediction of the results.

## 1. **Project Structure:**
* `data/`: Contains the original dataset `WineQT.csv`.
* `models/`: Contains the trained model (`model.joblib`) and the data scaler (`scaler.joblib`).
* `src/`: Folder with the source code in modules:
    * `eda.py`: Data cleaning and visualizations.
    * `entrenamiento.py`: Data processing, model training, and selection of the best model.
    * `prueba.py`: Loading the saved model, final metrics, and confusion matrix.

## 2. **Installation Requirements:**
To run this project, you need to have Python and pip installed.

## 3. **Running the Project:**
First, you must clone the repository. Once the directory path is '/WINE_REPO_SC', continue:

- Install the necessary libraries with the following command:
```pip install -r requirements.txt```

- You can then continue with the project flow:

### Data Analysis:
```python src/eda.py```

### Model Training and Selection:
```python src/entrenamiento.py```

### Final Test:
```python src/prueba.py```

## 4. **Expected Results:**
The Support Vector Machine (SVM) model was selected as the winner due to its superior ability to handle nonlinear relationships in chemical data, achieving higher accuracy compared to the Logistic Regression model.

**Developed By:** Samantha Castro
