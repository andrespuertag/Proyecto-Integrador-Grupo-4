# %%
# Libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
from scipy.stats import pearsonr, spearmanr
from scipy.stats import contingency
import pandas as pd
import numpy as np
import seaborn as sns
import csv
import joblib
import matplotlib.pyplot as plt
import subprocess

# %%
cmd = "aws s3 cp s3://taxdatalake/RAW/DCS_Item_Extract_Report C:\\Users\\user\\OneDrive\\EAFIT\\MAESTRÍA_EN_CIENCIA_DE_DATOS\\PROYECTO_INTEGRADOR\\BUCKET_PROJECT\\MODEL\\ --no-sign-request"

# Run the command
subprocess.run(cmd, shell=True)

# %%
path = r'C:\\Users\\user\\OneDrive\\EAFIT\\MAESTRÍA_EN_CIENCIA_DE_DATOS\\PROYECTO_INTEGRADOR\\BUCKET_PROJECT\\MODEL\\DCS_Item_Extract_Report'
df = pd.read_excel(path,header = 1)

# %%
df.head(3)

# %%
# Removed first column
df_Remove = df.iloc[0:, 1:]

# %%
df_Remove.head(3)

# %% [markdown]
# # <CENTER> **EDA** </CENTER>

# %%
n_tax_codes = df_Remove['VERTEX PROD ID'].unique()

# %%
Vertex_dist = df_Remove['VERTEX PROD ID'].value_counts()
Vertex_dist = Vertex_dist.sort_values(ascending=False)
print('Number of Vertex Categories unfiltered: ',len(Vertex_dist))
filtered_dist = Vertex_dist[Vertex_dist >= 50]
print('Number of filtered items with frequency higher than forty items: ',len(filtered_dist))
filtered_dist.head(10)

# %%
cum_dist_Vertex = Vertex_dist.sort_values()
cumulative_sum = Vertex_dist.cumsum()
cumulative_distribution = (cumulative_sum / df_Remove['VERTEX PROD ID'].count())*100
cumulative_distribution.head(52)

# %%
Vertex_dist = df_Remove['VERTEX PROD ID'].value_counts()
Vertex_dist = Vertex_dist.sort_values(ascending=False)
filtered_dist = Vertex_dist[Vertex_dist >= 100]
print('Total categories:',len(filtered_dist))
filtered_dist.head(10)

# %%
Vertex_dist.plot(kind='bar',figsize=[25,8])

# Set labels and title
plt.xlabel('Unique Values')
plt.ylabel('Count')
plt.title('Distribution of Unique Values in VERTEX PROD ID')

# Show the plot
plt.grid()
plt.show()

# %%
filtered_dist.plot(kind='bar',figsize=[25,8], color='Green')

print('Total number of unique values is: ', Vertex_dist.nunique())

# Set labels and title
plt.xlabel('Unique Values')
plt.ylabel('Count')
plt.title('Distribution of Unique Values in VERTEX PROD ID')

# Show the plot
plt.grid()
plt.show()

# %%
df_Remove.info()

# %%
# Removed blanks
df_V2 = df_Remove[df_Remove['PROD SUBSYSTEM DESC'].notnull()] #esta columna es la que tiene mas blanks
df_V2 = df_V2[df_V2['VERTEX PROD ID'].notnull()]
df_V2.info()

# %%
#Get the column names from column 2 onwards
column_names = df_V2.columns[1:]
column_names

# %%
# Create a dictionary to store label encoders
label_encode = {}
encoded_df = {}

# Fit and transform data, and store label encoders
for i in df_V2.iloc[0: ,1:27]:
    label_encode[i] = LabelEncoder()
    encoded_df[i] = label_encode[i].fit_transform(df_V2[i])

# Display the DataFrame
encoded_df = pd.DataFrame(encoded_df)
encoded_df

# %%
#Print encoded DF to visualize it
encoded_df.head(3)

# %%
#Exclude item number column
encoded_df2 = encoded_df.iloc[:, 1:]
columns_encoded = encoded_df2.columns
columns_encoded

# %%
#Transform columns into integer values as most of them are object types.
encoded_df2[columns_encoded] = encoded_df2[columns_encoded].astype(int)
encoded_df2.info()

# %%
# Calculate correlational matrix
correlation_matrix = encoded_df2.corr()

# %%
#filtro de la matrix
high_corr = correlation_matrix[correlation_matrix>=.50]

#plt.figure(figsize=(12,8))
sns.heatmap(high_corr, cmap='Reds')
plt.title('High Positive correlation')
plt.grid();

# %%
#filtro de la matrix
low_corr = correlation_matrix[correlation_matrix<=-0.40]

#plt.figure(figsize=(12,8))
sns.heatmap(low_corr, cmap='Blues')
plt.title('High Negative Correlation')
plt.grid();

# %% [markdown]
# # <CENTER> **RANDOM FOREST DE BASE** </CENTER>

# %%
X = encoded_df2.drop('VERTEX PROD ID', axis = 1)
y = encoded_df2['VERTEX PROD ID']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf = clf.fit(X_train, y_train)

# Realizar predicciones
y_pred = clf.predict(X_test)

# Calcular la precisión del clasificador
accuracy = balanced_accuracy_score(y_test, y_pred)
print(f"Precisión del clasificador: {accuracy:.2f}")

# Escribir el informe en una variable
reportRF = classification_report(y_test, y_pred, zero_division=float('NaN'))
print("\nReporte de clasificación:")
print(reportRF)
 
# Escribir el informe en un archivo
with open('ReportRF.txt', 'w') as file:
    file.write(reportRF)

# %% [markdown]
# ---
# ---
# # <CENTER>**REDISTRIBUCIÓN DE DATOS**</CENTER>

# %%
# Removed blanks
df_V2 = df_Remove[df_Remove['PROD SUBSYSTEM DESC'].notnull()] #esta columna es la que tiene mas blanks
df_V2 = df_V2[df_V2['VERTEX PROD ID'].notnull()]
df_V2.head(2)

# %%
value_counts = df_V2['VERTEX PROD ID'].value_counts()
mask = value_counts < 50
replacement_value = 'MA-DIY'
df_V2['VERTEX PROD ID'] = df_V2['VERTEX PROD ID'].replace(value_counts[mask].index, replacement_value)
df_V2.head(2)
#df_V2.info()

# %%
random_var = df_V2['VERTEX PROD ID'].value_counts()
random_var.plot(kind='bar',figsize=[25,8], color='orange')

# Set labels and title
plt.xlabel('Unique Values')
plt.ylabel('Count')
plt.title('Distribution of Unique Values in VERTEX PROD ID')

# Show the plot
plt.grid()
plt.show()

# %%
#Create a copy of the dataframe and transform NaN values to Missing for ordinal encoding
encoBalance_df = df_V2.copy()
encoBalance_df.fillna('MISS', inplace=True)

ordinal_encodeBal = {}
encoded_dfBal = pd.DataFrame()



# Fit and transform data, and store ordinal encoders
for col in encoBalance_df.iloc[:, 1:].columns:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=9000)
    encoded_dfBal[col] = encoder.fit_transform(encoBalance_df[[col]]).flatten()
    ordinal_encodeBal[col] = encoder

# %%
# Save all ordinal encoders in a single file
joblib.dump(ordinal_encodeBal, 'ordinal_encoders.joblib')

# %%
encoded_dfBal[columns_encoded] = encoded_dfBal[columns_encoded].astype(int)

# %% [markdown]
# ---
# --- 
# # <CENTER>**PCA**</CENTER>

# %%
#Defino el número de componentes que quiero conservar
pca = PCA(n_components=22)
Scaler = StandardScaler()

# %%
#Separo X y Y
X_Scaled = encoded_dfBal.drop('VERTEX PROD ID', axis=1)
y_Scaled = encoded_dfBal['VERTEX PROD ID']

#Escalo X
X_Scaled = Scaler.fit_transform(X_Scaled)

#Convierto los datos en un dataframe-
X_Scaled = pd.DataFrame(X_Scaled)

# %%
X_trainScl, X_testScl, y_trainScl, y_testScl = train_test_split(X_Scaled, y_Scaled, test_size=0.2, random_state=42, stratify=y_Scaled)

X_train_pca = pca.fit_transform(X_trainScl)
X_test_pca = pca.transform(X_testScl)

# %%
# Calculate the cumulative explained variance
cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()

# Find the number of components that explain at least 95% of the variance
n_components = np.argmax(cumulative_explained_variance >= 0.95) + 1

# Plot the cumulative explained variance
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.axvline(n_components, color='red', linestyle='--', label=f'{n_components} Componentes')
plt.xlabel('Número de componentes')
plt.ylabel(' Varianza Acumulada explicada')
plt.title('Varianza acumulada explicada por componente')
plt.legend()
plt.grid()
plt.show()


# %%
pca = PCA(n_components=17)

X_train_pca = pca.fit_transform(X_trainScl)
X_test_pca = pca.transform(X_testScl)

# %%
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_trainScl)

# %%
predicted_y = rf_model.predict(X_test_pca)

# Evaluo la precisión
accuracy = balanced_accuracy_score(y_testScl, predicted_y)
print(f"Accuracy: {accuracy}")

# Escribir el informe en una variable
PCA_RF = classification_report(y_testScl, predicted_y, zero_division=float('NaN'))
print("\nReporte de clasificación:")
print(PCA_RF)
 
# Escribir el informe en un archivo
with open('PCA_RF.txt', 'w') as file:
    file.write(PCA_RF)

# %% [markdown]
# Después de aplicar el PCA se obtiene un balanced accuracy de 0.94, sin embargo para obtener esta puntuación es necesario mantener 17 componentes, lo cual no hace una gran diferencia a nivel computacional en el tiempo de predicción o en el tiempo de ejecución del modelo. De igual manera un PCA con este conjunto de datos con solo 24 variables no es necesario. Y emplearlo en este caso de negocio implicaría hacer una transformación adicional sobre los datos cada vez que se quieran hacer predicciones. Dado todo lo anterior, concluimos que para el caso de negocio y en vista de la reducida cantidad de variables y el bajo beneficio que se obtendría de un proceso de reducción de dimensionalidad como este, aplicar PCA es innecesario y decidimos prescindir del uso del mismo.

# %% [markdown]
# ---
# --- 
# # <CENTER>**BALANCEO DE DATOS CON SMOTE**</CENTER>

# %%
df_without_null = encoded_dfBal.dropna()

# %%
X_SMOTE = df_without_null.drop('VERTEX PROD ID', axis=1)
y_SMOTE = df_without_null['VERTEX PROD ID']

# Split the data into training and testing sets
X_train_SM, X_test_SM, y_train_SM, y_test_SM = train_test_split(X_SMOTE, y_SMOTE, test_size=0.2, random_state=42)

# Apply SMOTE to the training data only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_SM, y_train_SM)

# Create a new balanced DataFrame
df_balanced_SM = pd.concat([pd.DataFrame(X_train_resampled, columns=X_SMOTE.columns), pd.Series(y_train_resampled, name='VERTEX PROD ID')], axis=1)

# %%
clf_balanced_data = RandomForestClassifier(n_estimators=100,random_state=42)
clf_balanced_data = clf_balanced_data.fit(X_train_resampled, y_train_resampled)

# Realizar predicciones
y_pred_sm = clf_balanced_data.predict(X_test_SM)

# Calcular la precisión del clasificador
accuracy = balanced_accuracy_score(y_test_SM, y_pred_sm)
print(f"Precisión del clasificador: {accuracy:.4f}")

# Escribir el informe en una variable
Smote_RF = classification_report(y_test_SM, y_pred_sm, zero_division=float('NaN'))
print("\nReporte de clasificación:")
print(Smote_RF)
 
# Escribir el informe en un archivo
with open('SMOTE_RF.txt', 'w') as file:
    file.write(Smote_RF)

# %% [markdown]
# ---
# --- 
# # <CENTER>**PROCESO SIN PCA - CONJUNTO DE DATOS TRANSFORMADO (53 CLASES)**</CENTER>

# %%
XBal = encoded_dfBal.drop('VERTEX PROD ID', axis = 1)
yBal = encoded_dfBal['VERTEX PROD ID']

X_trainBal, X_testBal, y_trainBal, y_testBal = train_test_split(XBal, yBal, test_size=0.2, random_state=42, stratify=yBal)

# %%
clfBal = RandomForestClassifier(n_estimators=100,random_state=42)
clfBal = clfBal.fit(X_trainBal, y_trainBal)

# Realizar predicciones
y_pred = clfBal.predict(X_testBal)

# Calcular la precisión del clasificador
accuracy = balanced_accuracy_score(y_testBal, y_pred)
print(f"Precisión del clasificador: {accuracy:.4f}")

# Escribir el informe en un archivo
reportclfBal = classification_report(y_testBal, y_pred, zero_division=float('NaN'))
with open('ReportclfBal.txt', 'w') as file:
    file.write(reportclfBal)

# %% [markdown]
# # <CENTER> **XGBOOST CON CONJUNTO DE DATOS TRANSFORMADO** </CENTER>

# %%
#XG Boost
model = xgb.XGBClassifier()
model.fit(X_trainBal, y_trainBal)

y_pred = model.predict(X_testBal)

accuracy = balanced_accuracy_score(y_testBal, y_pred)
print(f"Precisión del clasificador: {accuracy:.4f}")

# Escribir el informe en un archivo
reportXGBal = classification_report(y_testBal, y_pred, zero_division=float('NaN'))
with open('ReportXGBal.txt', 'w') as file:
    file.write(reportXGBal)

# %% [markdown]
# # <CENTER> **BAGGING BASADO EN ÁRBOLES - CONJUNTO DE DATOS TRANSFORMADO** </CENTER>

# %%
# Crear un clasificador base (en este caso, un árbol de decisión)
base_classifier = DecisionTreeClassifier(random_state=42)

# Crear el clasificador Bagging usando el clasificador base
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Entrenar el clasificador Bagging
bagging_classifier.fit(X_trainBal, y_trainBal)

# Realizar predicciones en el conjunto de prueba
predictions = bagging_classifier.predict(X_testBal)

# Calcular la precisión
accuracy = balanced_accuracy_score(y_testBal, predictions)
print(f"Accuracy: {accuracy:.4f}")

# Escribir el informe en un archivo
reportBaggingBal = classification_report(y_testBal, predictions, zero_division=float('NaN'))
with open('ReportBaggingBal.txt', 'w') as file:
    file.write(reportBaggingBal)

# %% [markdown]
# # <CENTER> **BAGGING BASADO EN ÁRBOLES - CON SMOTE** </CENTER>

# %%
# Crear un clasificador base (en este caso, un árbol de decisión)
base_classifier2 = DecisionTreeClassifier(random_state=42)

# Crear el clasificador Bagging usando el clasificador base
bagging_classifier2 = BaggingClassifier(base_classifier2, n_estimators=10, random_state=42)

# Entrenar el clasificador Bagging
bagging_classifier2.fit(X_train_SM, y_train_SM)

# Realizar predicciones en el conjunto de prueba
predictions2 = bagging_classifier2.predict(X_test_SM)

# Calcular la precisión
accuracy = balanced_accuracy_score(y_test_SM, predictions2)
print(f"Accuracy: {accuracy:.4f}")

# Escribir el informe en un archivo
reportBagging_SM = classification_report(y_test_SM, predictions2, zero_division=float('NaN'))
with open('reportBagging_SM.txt', 'w') as file:
    file.write(reportBagging_SM)

# %% [markdown]
# # <center> **VALIDACIÓN DEL MEJOR MODELO** </center>

# %%
# Random Forest - datos con todas las clases
ruta_archivo = 'ReportRF.txt'
MetricsRF = pd.read_csv(ruta_archivo, delimiter='\t')
 
def separa_col(dataframe):
    nuevas_columnas = dataframe[dataframe.columns[0]].str.split(expand=True)
# Asigna las nuevas columnas al DataFrame original
    dataframe['No'] = nuevas_columnas[0]
    dataframe['Precision'] = nuevas_columnas[1]
    dataframe['Recall'] = nuevas_columnas[2]
    dataframe['F1score'] = nuevas_columnas[3]
    dataframe['Support'] = nuevas_columnas[4]
 
# Elimina la columna original
    dataframe.drop(columns=[dataframe.columns[0]], inplace=True)
 
separa_col(MetricsRF)
 
MetricsRF = MetricsRF.iloc[:-3].astype(float)

# %%
# XGBoost - datos con la clase MA-DIY
ruta_archivo = 'ReportXGBal.txt'
MetricsXGBal = pd.read_csv(ruta_archivo, delimiter='\t')
 
separa_col(MetricsXGBal)
 
MetricsXGBal = MetricsXGBal.iloc[:-3].astype(float)

# %%
# Bagging - datos con la clase MA-DIY
ruta_archivo = 'ReportBaggingBal.txt'
MetricsBaggingBal = pd.read_csv(ruta_archivo, delimiter='\t')
 
separa_col(MetricsBaggingBal)
 
MetricsBaggingBal = MetricsBaggingBal.iloc[:-3].astype(float)

# %%
# Random Forest - datos con la clase MA-DIY
ruta_archivo = 'ReportclfBal.txt'
MetricsclfBal = pd.read_csv(ruta_archivo, delimiter='\t')
 
separa_col(MetricsclfBal)
 
MetricsclfBal = MetricsclfBal.iloc[:-3].astype(float)

# %%
# PCA - datos con la clase MA-DIY
ruta_archivo = 'PCA_RF.txt'
MetricsPCA = pd.read_csv(ruta_archivo, delimiter='\t')
 
separa_col(MetricsPCA)
 
MetricsPCA = MetricsPCA.iloc[:-3].astype(float)

# %%
# SMOTE - datos con la clase MA-DIY
ruta_archivo = 'SMOTE_RF.txt'
MetricsSmote = pd.read_csv(ruta_archivo, delimiter='\t')
 
separa_col(MetricsSmote)
 
MetricsSmote = MetricsSmote.iloc[:-3].astype(float)

# %%
# SMOTE - BAGGING
ruta_archivo = 'reportBagging_SM.txt'
MetricsSMOTE_BG = pd.read_csv(ruta_archivo, delimiter='\t')
 
separa_col(MetricsSMOTE_BG)
 
MetricsSMOTE_BG = MetricsSMOTE_BG.iloc[:-3].astype(float)

# %% [markdown]
# # **PRECISION SCORE**

# %%
PrecisionTodos = pd.concat([MetricsRF['Precision'],MetricsclfBal['Precision'],MetricsXGBal['Precision'],MetricsBaggingBal['Precision'],MetricsPCA['Precision'],MetricsSmote['Precision'], MetricsSMOTE_BG['Precision']],axis=1)
PrecisionTodos = PrecisionTodos.dropna()
PrecisionTodos.columns = ['RF','RF Bal','XG Boost','Bagging','PCA_RF', 'SMOTE_RF', 'SMOTE-BAGG']
 
# Cambiar la tipografía en Seaborn
sns.set(font='Times New Roman',  # Cambia la fuente del texto
        rc={
            'axes.labelsize': 12.5,  # Tamaño de la etiqueta de los ejes
            'xtick.labelsize': 12,  # Tamaño de las etiquetas del eje x
            'ytick.labelsize': 12  # Tamaño de las etiquetas del eje y
        })
sns.set_style("whitegrid")
# Crear el boxplot con seaborn
plt.figure(figsize=(10, 8))  # Tamaño del gráfico opcional
sns.violinplot(data=PrecisionTodos, linewidth=0.5, edgecolor='blue', palette='Blues')
plt.title('Comparación Precisión por modelos')
plt.xlabel('Modelos')
plt.ylabel('Precisión')
plt.grid(True)
plt.show()

# %%
PrecisionTodos.describe()

# %% [markdown]
# # **RECALL SCORE**

# %%
RecallTodos = pd.concat([MetricsRF['Recall'],MetricsclfBal['Recall'],MetricsXGBal['Recall'],MetricsBaggingBal['Recall'],MetricsPCA['Recall'],MetricsSmote['Recall'],MetricsSMOTE_BG['Recall']],axis=1)
RecallTodos = RecallTodos.dropna()
RecallTodos.columns = ['RF','RF Bal','XG Boost','Bagging','PCA_RF', 'SMOTE_RF', 'SMOTE-BAGG']
 
# Cambiar la tipografía en Seaborn
sns.set(font='Times New Roman',  # Cambia la fuente del texto
        rc={
            'axes.labelsize': 12.5,  # Tamaño de la etiqueta de los ejes
            'xtick.labelsize': 12,  # Tamaño de las etiquetas del eje x
            'ytick.labelsize': 12  # Tamaño de las etiquetas del eje y
        })
sns.set_style("whitegrid")
# Crear el boxplot con seaborn
plt.figure(figsize=(10, 8))  
sns.violinplot(data=RecallTodos, linewidth=0.5, edgecolor='blue', palette='Greens')
#sns.stripplot(data=PrecisionTodos, color="Black", size=4, jitter=True)  # scatter plot
plt.title('Comparación Recall por modelos')
plt.xlabel('Modelos')
plt.ylabel('Recall')
plt.grid(True)
plt.show()

# %%
RecallTodos.describe()

# %% [markdown]
# # **F1 SCORE**

# %%
F1Todos = pd.concat([MetricsRF['F1score'],MetricsclfBal['F1score'],MetricsXGBal['F1score'],MetricsBaggingBal['F1score'],MetricsPCA['F1score'],MetricsSmote['F1score'],MetricsSMOTE_BG['Recall']],axis=1)
F1Todos = F1Todos.dropna()
F1Todos.columns = ['RF','RF Bal','XG Boost','Bagging','PCA_RF', 'SMOTE_RF', 'SMOTE-BAGG']
 
# Cambiar la tipografía en Seaborn
sns.set(font='Times New Roman',  # Cambia la fuente del texto
        rc={
            'axes.labelsize': 12.5,  # Tamaño de la etiqueta de los ejes
            'xtick.labelsize': 12,  # Tamaño de las etiquetas del eje x
            'ytick.labelsize': 12  # Tamaño de las etiquetas del eje y
        })
sns.set_style("whitegrid")
# Crear el boxplot con seaborn
plt.figure(figsize=(10, 8))  
sns.violinplot(data=F1Todos, linewidth=0.5, edgecolor='blue', palette='Reds')
plt.title('Comparación F1-Score por modelos')
plt.xlabel('Modelos')
plt.ylabel('F1-Score')
plt.grid(True)
plt.show()

# %%
F1Todos.describe()

# %% [markdown]
# # <center> **CROSS VALIDATION RANDOMFOREST - SMOTE** </center>

# %%
'''# Definir los hiperparámetros a buscar
param_grid = {
    'n_estimators': [10, 50, 100, 250, 300],
    'max_depth': [None, 5, 10, 15, 20, 30],
    'min_samples_split': [2, 5, 7 ,10, 13],
    'min_samples_leaf': [1, 2, 4, 8]
}

# Realizar la búsqueda de hiperparámetros utilizando GridSearchCV
grid_search = GridSearchCV(estimator=clfBal, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_SM, y_train_SM)

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)'''

# %%
'''# Obtener el modelo con los mejores hiperparámetros y evaluar en el conjunto de prueba
best_rf = grid_search.best_estimator_
predictions = best_rf.predict(X_test_SM)
accuracy = balanced_accuracy_score(y_test_SM, predictions)
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")'''

# %% [markdown]
# # <center> **CROSS VALIDATION BAGGING - SMOTE (OPCIONAL)** </center>

# %%
'''base_estimator = DecisionTreeClassifier(random_state=42)

# Definir los hiperparámetros a buscar
param_grid = {
    'n_estimators': [10, 50, 100, 250, 300],
    'max_samples': [0.5, 0.7, 0.3, 1.0],  # Fracción de muestras usadas para cada iteración
    'max_features': [0.5, 0.7, 0.3, 1.0],  # Fracción de características usadas
    'bootstrap': [True, False],  # ¿Las muestras son tomadas con reemplazo?
    'bootstrap_features': [True, False]  # Las características son tomadas con reemplazo?
}

#Crear el clasificador
bagging_clf = BaggingClassifier(base_estimator=base_estimator, random_state=42)

# Realizar la búsqueda de hiperparámetros utilizando GridSearchCV
grid_search_bagging = GridSearchCV(estimator=bagging_clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_bagging.fit(X_train_SM, y_train_SM)

# Mostrar los mejores hiperparámetros encontrados
print("Mejores hiperparámetros encontrados:")
print(grid_search_bagging.best_params_)'''

# %%
'''# Obtener el modelo con los mejores hiperparámetros y evaluar en el conjunto de prueba
best_bagging = grid_search_bagging.best_estimator_
predictions_bagging = best_bagging.predict(X_test_SM)
accuracy = balanced_accuracy_score(y_test_SM, predictions_bagging)
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")'''

# %% [markdown]
# # <center> **ENTRENAMIENTO DEL MEJOR MODELO OBTENIDO** </center>

# %%
#Se usa todo el conjunto de datos remuestreado con SMOTE para evitar que las clases minoritarias estén por fuera de los datos de entrenamiento del modelo
Final = RandomForestClassifier(random_state=42, max_depth= 20, min_samples_leaf= 1, min_samples_split=2, n_estimators=250)
Final = Final.fit(X_SMOTE, y_SMOTE)

model_filename = r"Final_model.pkl"
joblib.dump(Final, model_filename)


