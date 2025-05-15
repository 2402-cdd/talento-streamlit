
import streamlit as st
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Cargar archivos
model = pickle.load(open('modelo_rf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))
vectores_designation = pickle.load(open('vectores_por_puesto.pkl', 'rb'))

# Lista de habilidades técnicas
habilidades = [
    'PYTHON', 'C++', 'JAVA', 'HADOOP', 'SCALA', 'FLASK', 'PANDAS', 'SPARK',
    'NUMPY', 'PHP', 'SQL', 'MYSQL', 'CSS', 'MONGODB', 'NLTK', 'TENSORFLOW',
    'LINUX', 'RUBY', 'JAVASCRIPT', 'DJANGO', 'REACT', 'REACTJS', 'AI', 'UI',
    'TABLEAU', 'NODEJS', 'EXCEL', 'POWER BI', 'SELENIUM', 'HTML', 'ML'
]

st.title("🔍 Conectando Talento y Oportunidad")
st.markdown("Simula tu perfil y descubre el **nivel estimado** y los **puestos que más coinciden** con tus habilidades.")

# --- Selección de habilidades ---
st.subheader("1️⃣ Selecciona tus habilidades técnicas")

inputs = []
for skill in habilidades:
    inputs.append(1 if st.checkbox(skill) else 0)

# --- Inputs adicionales ---
st.subheader("2️⃣ Completa tu información de perfil")

designation = st.selectbox("🧑‍💼 Título actual o deseado", encoders['Designation'].classes_)
location = st.selectbox("🌎 Ubicación", encoders['Location'].classes_)
industry = st.selectbox("🏭 Industria", encoders['Industry'].classes_)

designation_enc = encoders['Designation'].transform([designation])[0]
location_enc = encoders['Location'].transform([location])[0]
industry_enc = encoders['Industry'].transform([industry])[0]

# Agregar inputs adicionales al vector
inputs.extend([designation_enc, location_enc, industry_enc])

X_input = np.array(inputs).reshape(1, -1)
X_scaled = scaler.transform(X_input)

# --- Predicción de nivel y sugerencia de Designations ---
if st.button("🔮 Predecir nivel y sugerir puestos"):
    pred = model.predict(X_scaled)
    niveles = {0: 'Junior', 1: 'Mid-Level', 2: 'Senior'}
    resultado = niveles.get(pred[0], "Desconocido")
    
    st.success(f"🎯 Nivel estimado: **{resultado}**")

    # --- Recomendación de Designations similares ---
    st.markdown("### 🤝 Puestos que mejor coinciden con tus habilidades:")
    
    # Solo usamos las primeras 31 columnas (habilidades) para la similitud
    user_skills = np.array(inputs[:31]).reshape(1, -1)

    similarity_scores = []
    for puesto, vector in vectores_designation.items():
        vector_values = np.array([vector.get(skill, 0) for skill in habilidades]).reshape(1, -1)
        similarity = cosine_similarity(user_skills, vector_values)[0][0]
        similarity_scores.append((puesto, round(similarity, 3)))

    # Ordenar y mostrar top 3
    top_matches = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]

    with st.expander("🔎 Ver sugerencias de Designations compatibles"):
        for puesto, sim in top_matches:
            st.markdown(f"- ✅ **{puesto}** (similitud: `{sim}`)")
