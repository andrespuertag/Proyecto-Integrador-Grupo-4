# Proyecto-Integrador-Grupo-4
APLICACIÓN DE UN MODELO DE CLASIFICACIÓN DE CATEGORÍAS FISCALES DE UNA EMPRESA DE MANUFACTURACIÓN Y COMERCIALIZACIÓN DE DISPOSITIVOS BIOMÉDICOS
-------------------------------------------

Los impuestos indirectos son según el Internal Revenue Service of the United States (IRS)  “un impuesto que se puede dirigir hacia los otros, tal como el impuesto de propiedad sobre el negocio”. Esto implica, la asignación de impuestos sobre cosas como propiedad raíz, automóviles, inventario de la compañía y maquinaria o servicios de software propios de la empresa. Según EY, una de las más grandes consultoras a nivel mundial en la materia. esto generaría otros impuestos como: impuesto al valor añadido (IVA), Impuesto sobre bienes y servicios, impuesto sobre las ventas, impuestos sobre el uso de inventario y maquinaria propiedad de la compañía . En este contexto, en aras de proporcionar una categoría fiscal a cada uno de los dispositivos biomédicos contenidos en el conjunto de datos de este proyecto, se propone evaluar modelos de clasificación para definir el clasificador que mejor asigne la clase o categoría de impuesto, a partir de la comprensión de patrones presentes en un conjunto de características asignadas a cada dispositivo. 

En el ámbito del Machine Learning (ML), los problemas de clasificación son esenciales y tienen un impacto significativo en diversas disciplinas, desde la automatización de decisiones hasta la medicina. La clasificación implica prever respuestas cualitativas asignando observaciones a categorías específicas. Diversas estrategias de ML abordan problemas de clasificación, como tareas binarias, de múltiples clases, de múltiples etiquetas y jerárquicas. Entre las técnicas de clasificación, el uso de PCA como preprocesamiento destaca por su capacidad para reducir la dimensionalidad y eliminar la correlación entre atributos. Además, la clasificación de conjuntos de datos no balanceados presenta desafíos, abordados por enfoques a nivel de datos, como SMOTE, que genera instancias sintéticas para contrarrestar la falta de representación de la clase minoritaria. A nivel de algoritmo, métodos de ensamble como Random Forest y Boosting se destacan por mejorar la precisión y generalización en conjuntos de datos no balanceados. La técnica de Bagging, mediante el muestreo con reemplazo, también contribuye a reducir la varianza y mejorar la estabilidad de los modelos. En general, la diversidad de estrategias y enfoques en el ámbito de la clasificación en ML enriquece la capacidad para abordar con éxito la complejidad de los desafíos en el análisis de datos y la toma de decisiones.

De esta manera, el objetivo principal de este proyecto es desarrollar un modelo de clasificación efectivo en el ámbito del Machine Learning para asignar categorías fiscales a dispositivos biomédicos. Este objetivo surge de la necesidad de proporcionar una clasificación fiscal a cada dispositivo biomédico en el conjunto de datos del proyecto. Se propone evaluar modelos de clasificación que utilicen técnicas como PCA para reducir la dimensionalidad y abordar la correlación entre atributos. Además, se abordarán desafíos específicos, como conjuntos de datos no balanceados, mediante enfoques como SMOTE. Se considerarán métodos de ensamble como Random Forest y Boosting para mejorar la precisión y generalización en conjuntos de datos desbalanceados. En resumen, el objetivo es utilizar estrategias y enfoques avanzados en el ámbito de la clasificación en Machine Learning para lograr una asignación precisa de categorías fiscales a dispositivos biomédicos, contribuyendo así a la eficacia en el análisis de datos y la toma de decisiones en este contexto, con el fin de abolir el error humano y por ende los riesgos fiscales, legales y financieros.

El proyecto consta de dos momentos, el de Entrenamiento del Modelo, y Limpieza, Transformación de Datos y Aplicación del Modelo.

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



