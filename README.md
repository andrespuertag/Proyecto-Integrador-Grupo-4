# Proyecto-Integrador-Grupo-4
APLICACIÓN DE UN MODELO DE CLASIFICACIÓN DE CATEGORÍAS FISCALES DE UNA EMPRESA DE MANUFACTURACIÓN Y COMERCIALIZACIÓN DE DISPOSITIVOS BIOMÉDICOS
-------------------------------------------
# Entrenamiento del modelo

Descripción General

El objetivo de este proyecto es la implementación de una estrategia de aprendizaje automático para clasificación de categorías fiscales de una empresa de dispositivos biomédicos. Cubre diversos aspectos del ciclo de vida del aprendizaje automático, incluyendo la carga de datos, preprocesamiento, análisis exploratorio de datos (EDA), entrenamiento y evaluación de modelos, así como comparación de modelos. El objetivo principal es predecir la variable 'VERTEX PROD ID' basándose en las características del conjunto de datos.

Requisitos Previos

Antes de ejecutar el script, se debe tener instaladas las bibliotecas y dependencias de Python necesarias, son:

 

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

-	Reports: Contiene informes de clasificación para cada modelo.
-	Model: Almacena el modelo final de aprendizaje automático entrenado.
-	Visualización: Incluye visualizaciones de la distribución de datos y el rendimiento del modelo.

-------------------------------------------

#	Limpieza, Transformación de Datos y Aplicación del Modelo

Descripción General

En este segundo momento del proyecto se describe el flujo de limpieza, transformación de datos y aplicación de un modelo para predecir 'VERTEX PROD ID'. A continuación, se presenta un resumen paso a paso de lo que realiza el script.

Descarga del Archivo desde S3 (Ingesta)

El script inicia descargando directamente desde un Bucket de S3 utilizando el comando AWS CLI. La descarga se realiza en la carpeta local definida.

Limpieza de Datos

Se define una función cleaning que carga un archivo Excel, elimina columnas innecesarias y filtra filas con valores no nulos en columnas específicas. El conjunto de datos resultante se almacena en el DataFrame df.

Exportación del DataFrame a la Zona Trusted

Se identifican las observaciones que se desean predecir (aquellas con 'VERTEX PROD ID' vacías) y se exportan a un archivo CSV en la carpeta Trusted.

Aplicación del Label Encoder

Se carga un modelo de codificación previamente guardado (ordinal_encoders.joblib). Se define una función encode que aplica el encoder a las columnas del DataFrame.

Transformación de Datos a Representaciones Numéricas

Se realiza la transformación de los datos a sus representaciones numéricas, eliminando la columna 'VERTEX PROD ID' y mostrando el DataFrame resultante.

Extracción de Modelos

Se cargan diferentes modelos previamente entrenados (trained_model_rf.pkl, trained_model_XG.pkl, trained_model_Bagging.pkl, Final_model.pkl). Se selecciona uno de ellos para hacer predicciones.

Hacer Predicciones

Se utiliza el modelo seleccionado para hacer predicciones sobre el DataFrame transformado. Se implementa una función decode para invertir la codificación y obtener las etiquetas originales.

Exportación de Resultados a la Zona Refined

Se exportan los resultados de las predicciones y las observaciones no identificadas a archivos CSV en la carpeta Refined.

Organización del Proyecto

-	1_RAW: Contiene el archivo descargado desde S3.
-	2_TRUSTED: Almacena el DataFrame limpio y las observaciones a predecir.
-	3_REFINED: Contiene los resultados de las predicciones y las observaciones no identificadas.
-	ordinal_encoders.joblib: Modelo de Label Encoder guardado.



