import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo
modelo = joblib.load('modelo_iris.joblib')

# Título
st.title("Clasificador de Especies Iris 🌸")
st.info("Jesus Alvarado")
st.write("Ingrese los datos manualmente o cargue un archivo CSV para predecir la especie de Iris.")

# Sidebar con opciones
opcion = st.radio("Selecciona una forma de ingresar datos:", ['Manual', 'CSV'])

if opcion == 'Manual':
    st.subheader("Ingresar valores manuales")

    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1, value=5.1)
    sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1, value=3.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1, value=1.4)
    petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1, value=0.2)

    datos = pd.DataFrame([{
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }])

elif opcion == 'CSV':
    st.subheader("Subir archivo CSV")
    archivo = st.file_uploader("Selecciona un archivo CSV", type="csv")
    
    if archivo is not None:
        datos = pd.read_csv(archivo)
        st.write("Datos cargados:")
        st.dataframe(datos)
    else:
        datos = None

# Botón para hacer predicción
if st.button("Predecir"):
    if datos is not None:
        predicciones = modelo.predict(datos)
        # Traducir a nombres
        especies = ['setosa', 'versicolor', 'virginica']
        nombres_pred = [especies[i] for i in predicciones]

        st.subheader("Predicciones:")
        st.write(nombres_pred)
    else:
        st.warning("No se han ingresado datos.")
