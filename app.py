
import streamlit as st
import numpy as np
import pickle

model = pickle.load(open('modelo_rf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

habilidades = [
    'PYTHON', 'C++', 'JAVA', 'HADOOP', 'SCALA', 'FLASK', 'PANDAS', 'SPARK',
    'NUMPY', 'PHP', 'SQL', 'MYSQL', 'CSS', 'MONGODB', 'NLTK', 'TENSORFLOW',
    'LINUX', 'RUBY', 'JAVASCRIPT', 'DJANGO', 'REACT', 'REACTJS', 'AI', 'UI',
    'TABLEAU', 'NODEJS', 'EXCEL', 'POWER BI', 'SELENIUM', 'HTML', 'ML'
]

st.title("🔍 Conectando Talento y Oportunidad")
st.subheader("Simula tu perfil y descubre el nivel de puesto que podrías alcanzar")

st.write("Selecciona tus habilidades técnicas para obtener una predicción del nivel de vacante al que podrías aplicar:")

inputs = []
for skill in habilidades:
    selected = st.checkbox(skill)
    inputs.append(1 if selected else 0)

if st.button("Predecir nivel de vacante"):
    X_input = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)

    niveles = {0: 'Junior', 1: 'Mid-Level', 2: 'Senior'}
    resultado = niveles.get(pred[0], "Nivel desconocido")

    st.success(f"✅ Nivel estimado: **{resultado}**")
