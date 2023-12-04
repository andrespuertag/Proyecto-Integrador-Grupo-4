ReadMe del Script de Limpieza, Transformación de Datos y Aplicación del Modelo

Descripción General
Este script de Python aborda el flujo de limpieza, transformación de datos y aplicación de un modelo para predecir 'VERTEX PROD ID'. A continuación, se presenta un resumen paso a paso de lo que realiza el script.

Descarga del Archivo desde S3 (Ingesta)
El script inicia descargando directamente desde un bucket de S3 utilizando el comando AWS CLI. La descarga se realiza en la carpeta local definida.

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
1_RAW: Contiene el archivo descargado desde S3.
2_TRUSTED: Almacena el DataFrame limpio y las observaciones a predecir.
3_REFINED: Contiene los resultados de las predicciones y las observaciones no identificadas.
ordinal_encoders.joblib: Modelo de Label Encoder guardado.
Conclusiones
Este script es parte integral del proceso de predicción de 'VERTEX PROD ID', cubriendo desde la descarga de datos hasta la aplicación del modelo y exportación de resultados. Facilita la integración de nuevos datos y la aplicación del modelo entrenado para realizar predicciones.