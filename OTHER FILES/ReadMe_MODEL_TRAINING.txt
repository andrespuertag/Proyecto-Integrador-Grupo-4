ReadMe del Proyecto de Aprendizaje Automático

Descripción General

Este script de Python forma parte de un proyecto de aprendizaje automático para clasificación de categorías fiscales de una empresa de dispositivos biomédicos. Cubre diversos aspectos del ciclo de vida del aprendizaje automático, incluyendo la carga de datos, preprocesamiento, análisis exploratorio de datos (EDA), entrenamiento y evaluación de modelos, así como comparación de modelos. El objetivo principal es predecir la variable 'VERTEX PROD ID' basándose en las características del conjunto de datos.

Requisitos Previos

Antes de ejecutar el script, se debe tener instaladas las bibliotecas y dependencias de Python necesarias. Puedes utilizar los siguientes comandos para instalarlas:

bash

pip install scikit-learn xgboost imbalanced-learn pandas numpy seaborn matplotlib joblib

Carga de Datos

El script comienza cargando datos desde un depósito de AWS S3. Los datos se leen desde un archivo Excel y se eliminan las columnas innecesarias.

Análisis Exploratorio de Datos (EDA)

Se realiza un análisis exploratorio de datos para comprender el conjunto de datos. Esto incluye analizar la distribución de valores únicos en la columna 'VERTEX PROD ID' y visualizar los datos.

Preprocesamiento de Datos

El preprocesamiento de datos implica manejar los valores faltantes, codificación de etiquetas y transformar columnas de tipo objeto en valores enteros. Además, el script explora correlaciones entre características utilizando una matriz de correlación.

Transformación de Datos y Reducción de Dimensionalidad (PCA)

Se aplica el Análisis de Componentes Principales (PCA) para la reducción de dimensionalidad. Sin embargo, el script concluye que PCA no es necesario para este conjunto de datos específico debido a su baja dimensionalidad.

Balanceo de Datos con SMOTE

El script balancea los datos utilizando la Técnica de Sobremuestreo Minoritario Sintético (SMOTE) para abordar problemas de desequilibrio de clases.

Entrenamiento y Evaluación de Modelos

Se entrenan y evalúan varios modelos de aprendizaje automático, incluyendo Bosques Aleatorios (Random Forest), XGBoost y Bagging. El script realiza un análisis detallado del rendimiento del modelo, incluyendo precisión, recall y F1-score. También se aplica validación cruzada para ajustar hiperparámetros utilizando GridSearchCV.

Comparación de Modelos y Visualizaciones

El script genera visualizaciones que comparan la precisión, recall y F1-score de diferentes modelos. Incluye gráficos de violín para cada métrica, proporcionando información sobre la distribución de puntuaciones para cada modelo.

Entrenamiento del Modelo Final

El modelo final de aprendizaje automático (Bosques Aleatorios) se entrena en todo el conjunto de datos, incluyendo los datos balanceados obtenidos mediante SMOTE. El modelo entrenado se guarda para uso futuro.

Organización del Proyecto

Reports: Contiene informes de clasificación para cada modelo.
Model: Almacena el modelo final de aprendizaje automático entrenado.
Visualization: Incluye visualizaciones de la distribución de datos y el rendimiento del modelo.

Conclusión

Este script sirve como una guía integral para construir, entrenar y evaluar modelos de aprendizaje automático para predecir 'VERTEX PROD ID'. Proporciona flexibilidad para futuras exploraciones y mejoras en el modelo.