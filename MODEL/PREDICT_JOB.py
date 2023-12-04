# %% [markdown]
# # <Center> **FLUJO DE LIMPIEZA, TRANSFORMACIÓN DE DATOS Y APLICACIÓN DEL MODELO** </Center> 

# %%
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import subprocess

# %% [markdown]
# ---
# ---
# # <Center> **DESCARGAR EL ARCHIVO A LA ZONA RAW LOCAL DESDE S3 (INGEST)** </Center> 

# %%
#Este paso se ejecuta a través de shell en donde se hace una descarga directa desde el bucket de S3 a través de IP's permitidas para acceder al bucket (Revisar documentación)
cmd = "aws s3 cp s3://taxdatalake/RAW/DCS_Item_Extract_Report C:\\Users\\user\\OneDrive\\EAFIT\\MAESTRÍA_EN_CIENCIA_DE_DATOS\\PROYECTO_INTEGRADOR\\BUCKET_PROJECT\\1_RAW\\ --no-sign-request"

# Correr el comando
subprocess.run(cmd, shell=True)

# %%
#Defino la ruta en donde se almacenó el archivo de S3 para recuperar los datos.
path = r'C:\\Users\\user\\OneDrive\\EAFIT\\MAESTRÍA_EN_CIENCIA_DE_DATOS\\PROYECTO_INTEGRADOR\\BUCKET_PROJECT\\1_RAW\\DCS_Item_Extract_Report'

# %%
#Función para realizar la limpieza de los datos desde la ruta definida

def cleaning(path):
    df = pd.read_excel(path,header = 1)
    df = df.iloc[0:, 1:]
    columns = df.iloc[0: ,2:20].columns
    for column in columns:
        df = df[df[column].notnull()]
    return(df)

df = cleaning(path)

# %% [markdown]
# # <CENTER> **EXPORTAR EL DATAFRAME A LA ZONA TRUSTED** </CENTER>
# 
# ---
# ---

# %%
#Indico cuáles son las observaciones que quiero predecir a partir de mi conjunto de datos (Solo VERTEX PROD ID vacías)
target = df[df['VERTEX PROD ID'].isnull()]
target.fillna('MISS', inplace=True)
target['VERTEX PROD ID'] = target['VERTEX PROD ID'].replace('MISS', '')

# %%
target.to_csv(r'C:\Users\user\OneDrive\EAFIT\MAESTRÍA_EN_CIENCIA_DE_DATOS\PROYECTO_INTEGRADOR\BUCKET_PROJECT\2_TRUSTED\DataFrame.csv', index_label='Index')

# %% [markdown]
# # <CENTER> **APLICAR EL LABEL ENCODER** </CENTER>
# 
# ---
# ---

# %%
LabelEncode = joblib.load('ordinal_encoders.joblib')

# %%
def encode(df, encoding_table):
    encoded_df = df.copy()

    for col, encoder in encoding_table.items():
        if col in encoded_df.columns:  # Check if the column exists in the DataFrame
            # Handle NaN values before transforming
            encoded_df[col].fillna('MISS', inplace=True)  # Adjust 'missing' as needed

            # Transform the column using the encoder
            encoded_df[col] = encoder.transform(encoded_df[[col]]).flatten()

    return encoded_df

# %%
target_noitem = target.iloc[:,1:]

# Use the encode function to transform your data
encoded_df = encode(target_noitem, LabelEncode)

# %% [markdown]
# A partir de este punto, debemos transformar los datos a cada una de sus representaciones numéricas.

# %%
encoded_df = encoded_df.drop('VERTEX PROD ID',axis=1)
encoded_df.tail(1)

# %% [markdown]
# # <CENTER> **EXTRAER LOS MODELOS** </CENTER>

# %%
#model = joblib.load(r"trained_model_rf.pkl")
#model = joblib.load(r'trained_model_XG.pkl')
#model = joblib.load(r'trained_model_Bagging.pkl')
model = joblib.load(r'Final_model.pkl')

# %%
#encoded_df['VERTEX PROD ID'] = random_forest_model.predict(encoded_df2)
encoded_df['VERTEX PROD ID'] = model.predict(encoded_df)

# %%
def decode(df, encoding_table):
    columns = df.columns
    for col in columns:
        df.loc[:, col] = encoding_table[col].inverse_transform(df[col].values.reshape(-1, 1))
    return df

# %%
encoded_df.head(3)

# %% [markdown]
# # <CENTER> **HACER PREDICCIONES** </CENTER>
# 
# ---
# ---

# %%
predicted_y = decode(encoded_df,LabelEncode)

# %%
predictions = pd.merge(target.iloc[:, :1], predicted_y, left_index=True, right_index=True)

# %%
#Separo las X no identificadas y que necesitan más información puesto que el modelo aún no las conoce.
New_features = predictions[predictions.isna().any(axis=1)]

# %%
#Me quedo solo con las predicciones que tienen datos completos y que serán clasificadas
predictions = predictions[predictions.notna().all(axis=1)]

# %% [markdown]
# # <CENTER> **EXPORTO MIS RESULTADOS A LA ZONA REFINED** </CENTER>

# %%
New_features.to_csv(r'C:\Users\user\OneDrive\EAFIT\MAESTRÍA_EN_CIENCIA_DE_DATOS\PROYECTO_INTEGRADOR\BUCKET_PROJECT\3_REFINED\New_features.csv', index_label='Index')

# %%
predictions.to_csv(r'C:\Users\user\OneDrive\EAFIT\MAESTRÍA_EN_CIENCIA_DE_DATOS\PROYECTO_INTEGRADOR\BUCKET_PROJECT\3_REFINED\Predictions.csv', index_label='Index')


