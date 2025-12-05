import streamlit as st
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
CHROMA_COLLECTION_NAME = "places"

@st.cache_resource
def load_resources():
    """Carga el modelo de embeddings y el cliente ChromaDB una sola vez."""
    try:
        st.write("Cargando modelo de Embeddings (pesado, solo una vez)...")
        from sentence_transformers import SentenceTransformer
        EMBEDDING_MODEL = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
        st.write("Modelo de Embeddings cargado.")

        st.write("Cargando cliente ChromaDB...")
        import chromadb
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        st.write("ChromaDB cargado y colección lista.")
        
        return EMBEDDING_MODEL, collection
    except Exception as e:
        st.error(f"Error al cargar recursos esenciales (Embeddings o DB): {e}")
        st.stop()
        

EMBEDDING_MODEL, chroma_collection = load_resources()


def get_embeddings(text: str) -> np.ndarray:
    embeddings = EMBEDDING_MODEL.encode([text])[0]
    return embeddings

def generar_user_embedding(survey_dict: dict) -> np.ndarray:
    """Función simulada o importada para generar el embedding del usuario con LLM."""
    try:
        st.info("Generando frase de usuario con LLM...")
        
        from openai import OpenAI
        OPENROUTER_API_KEY = "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"
        LLM_MODEL = "qwen/qwen3-235b-a22b-2507"

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )

        prompt = f"""En base a la encuesta respondida por un usuario turista interesado en recomendaciones de lugares para visitar en lima , 
    genera una frase corta que resuma sus intereses y preferencias de viaje. 
    por ejemplo:
    encuesta: "categorias_interesadas": playa , historia , "Modo de viaje": solo, "Presupuesto": medio
    resultado
    “Quiero lugares históricos, tranquilos, para ir solo, con buena vista al mar”
    IMPORTANTE: no digas el nombre propio de un lugar como Lagranja Azul solo puedes mencionar nombnres genericos como restaurant o polleria etc etc. RESPONDE UNICAMENTE CON LA FRASE, SIN EXPLICAR NADA MÁS.

    Aqui esta la encuesta:
        
        Encuesta: {json.dumps(survey_dict)}
        """
        
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "text"},
        )
        result = completion.choices[0].message.content.strip()
        st.success(f"Frase generada: '{result}'")

        
        user_embedding = get_embeddings(result)
        return user_embedding
    except Exception as e:
        st.error(f"Error al generar embedding del usuario con LLM: {e}")
        return None
def recomendar_lugares_optimizada(
    user_survey_dict: dict, 
    collection, 
    n_recomendaciones_finales: int = 5,
    n_resultados_chroma: int = 50
) -> List[Dict[str, Any]]:
    """Función de inferencia adaptada para usar los recursos cacheados."""
    import numpy as np
    
    user_embedding = generar_user_embedding(user_survey_dict)
    
    if user_embedding is None:
        return []

    resultados = collection.query(
        query_embeddings=[user_embedding.tolist()],
        n_results=n_resultados_chroma,
    )
    
    recomendaciones_unicas = []
    archivos_ya_vistos = set()
    
    for i in range(len(resultados['metadatas'][0])):
        metadata = resultados['metadatas'][0][i]
        
        distance = resultados['distances'][0][i] 
        source_file = metadata.get('source_file')
        
        if source_file and source_file not in archivos_ya_vistos:
            
            lugar = {
                "title": metadata.get('title', 'Título no disponible'),
                "description": metadata.get('descriptions', 'Descripción no disponible'),
                "similarity_score": distance 
            }
            
            recomendaciones_unicas.append(lugar)
            archivos_ya_vistos.add(source_file)

            if len(recomendaciones_unicas) >= n_recomendaciones_finales:
                break
    
    recomendaciones_finales = [
        {"title": rec["title"], "description": rec["description"]}
        for rec in recomendaciones_unicas
    ]
    
    return recomendaciones_finales
st.set_page_config(
    page_title="Sistema de Recomendación Turística",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📍 Sistema de Recomendación de Lugares Turísticos")
st.markdown("---")

with st.form("user_survey_form"):
    st.header("1. Cuéntanos sobre tu viaje")
    categorias = st.multiselect(
        "¿Qué tipo de lugares te interesan?",
        options=["Historia", "Cultura", "Gastronomía", "Playa", "Naturaleza", "Aventura", "Arte Moderno"],
        default=["Historia", "Cultura"],
        key="categorias_interesadas"
    )
    modo = st.radio(
        "¿Con quién viajas?",
        options=["Solo", "Pareja", "Familia", "Amigos", "Negocios"],
        key="modo_viaje"
    )
    presupuesto = st.select_slider(
        "¿Cuál es tu presupuesto?",
        options=["Bajo", "Medio", "Medio-Alto", "Alto"],
        value="Medio",
        key="presupuesto"
    )

    submitted = st.form_submit_button("✨ Obtener Recomendaciones")


if submitted:
    st.markdown("---")
    
    user_survey_dict = {
        "categorias_interesadas": categorias,
        "modo_viaje": modo,
        "presupuesto": presupuesto,
    }

    st.sidebar.subheader("Encuesta enviada:")
    st.sidebar.json(user_survey_dict)

    with st.spinner("Buscando las mejores opciones para ti..."):
        recomendaciones = recomendar_lugares_optimizada(
            user_survey_dict=user_survey_dict,
            collection=chroma_collection,
            n_recomendaciones_finales=10,
            n_resultados_chroma=40 
        )

    if recomendaciones:
        for idx, lugar in enumerate(recomendaciones, 1):
            st.success(f"Recomendación #{idx}")
            st.subheader(lugar['title'])
            st.write(lugar['description'])
            st.markdown("---")
    else:
        st.warning("No se pudieron encontrar recomendaciones. Asegúrate de que ChromaDB esté cargada correctamente y la clave de OpenRouter sea válida.")