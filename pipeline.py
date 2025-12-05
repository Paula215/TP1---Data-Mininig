import os
import json
import pandas as pd
import datetime
from pathlib import Path
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

OPENROUTER_API_KEY = ""
LLM_MODEL = "google/gemini-2.5-flash"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
CHROMA_COLLECTION_NAME = "places"

EMBEDDING_MODEL = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)

class Sentiment(BaseModel):
    overall: str = Field(..., description="Etiqueta de sentimiento: positive | negative | neutral")
    score: float = Field(..., description="Valor continuo del sentimiento (-1 a 1 o 0 a 1)")

class Review(BaseModel):
    title: str
    text: str
    rating: Optional[int]
    date_original: str
    year: Optional[int]
    month: str
    visit_category: str
    language: str
    sentiment: Sentiment
    tokens_count: int
    keywords: List[str]

LLM_PARSE_PROMPT = """<prompt>
  <instruction>
    Eros un analista de reseñas de viajes. Tu tarea es analizar cuidadosamente el contenido de la reseña (título, texto, fecha, rating, etc.) y producir un JSON estructurado que siga estrictamente el esquema definido.
  </instruction>

  <reasoning>
    <step>
      1. Identifica un <review_id>. Si no se proporciona, genera un identificador hash corto o incremental. (Nota: Este campo se añadirá fuera del LLM en el pipeline).
    </step>
    <step>
      2. Limpia el <title> de cualquier HTML o caracteres extraños. Si no existe, devuelve un string vacío "".
    </step>
    <step>
      3. Limpia el <text> eliminando HTML. Si está vacío o faltante, devuelve "".
    </step>
    <step>
      4. Lee el <rating>. Asegúrate de que sea un número entre 1 y 5. Si no existe, devuelve null.
    </step>
    <step>
      5. Conserva la fecha original en <date_original>.
          - Extrae <year> como entero. Si no existe, devuelve null.
          - Extrae <month> en inglés con nombre completo (ej: "November"). Si no existe, devuelve "".
          - Busca en la fecha alguna categoría oculta (ej. "Business", "Friends", "Solo", "Family", "Couple").
            Si no aparece, devuelve "Unknown".
    </step>
    <step>
      6. Detecta el <language> del texto principal (ej: "en" para inglés, "es" para español).
          Si no es posible, devuelve "".
    </step>
    <step>
      7. Realiza análisis de sentimiento:
          - <sentiment.overall>: clasifica como "positive", "negative" o "neutral".
            Si no puedes decidir, devuelve "neutral".
          - <sentiment.score>: asigna un valor numérico continuo entre -1 y 1
            (ej. positivo alto ≈ 0.8, negativo fuerte ≈ -0.8, neutral ≈ 0.0).
    </step>
    <step>
      8. Calcula <tokens_count> como la cantidad aproximada de palabras del texto.
          Si el texto está vacío, devuelve 0.
    </step>
    <step>
      9. Extrae <keywords> como una lista de 3–10 palabras clave más relevantes del texto.
          Usa palabras en minúsculas, sin acentos ni símbolos.
          Si no existen keywords útiles, devuelve una lista vacía [].
    </step>
  </reasoning>

  <format>
    La salida debe ser EXCLUSIVAMENTE un JSON válido, sin texto adicional, que cumpla exactamente el siguiente esquema:

    {
      "title": "string",
      "text": "string",
      "rating": int | null,
      "date_original": "string",
      "year": int | null,
      "month": "string",
      "visit_category": "Business | Friends | Solo | Family | Couple | Unknown",
      "language": "string",
      "sentiment": {
        "overall": "positive | negative | neutral",
        "score": float
      },
      "tokens_count": int,
      "keywords": ["string", ...]
    }
  </format>
  <important> solo devuelva el json sin texto adicional ni '''json del formato markdown </important>
</prompt>"""


def step_1_llm_parse_reviews(folder_path="tripadvisor_extractions", output_csv="reviews_structured.csv"):
    """
    Etapa 1: Procesa archivos JSON brutos, usa el LLM para estructurar reseñas
    y guarda el resultado consolidado en un archivo CSV.
    """
    print("--- ⚙️ Iniciando Paso 1: Extracción y Parsing con LLM ---")
    df_reviews = pd.DataFrame()
    review_counter = 1

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"❌ Error cargando {filename}: {e}")
                continue

        reviews = data.get("reviews", [])
        for review in reviews:
            try:
                completion = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": LLM_PARSE_PROMPT},
                        {"role": "user", "content": json.dumps(review, ensure_ascii=False)}
                    ],
                    response_format={
                        "type": "json_object",
                        "json_schema": Review.model_json_schema()
                    },
                )

                result = json.loads(completion.choices[0].message.content)
                
                result["id"] = review_counter
                new_row = pd.json_normalize([result]).rename(columns={
                    'sentiment.overall': 'sentiment_overall',
                    'sentiment.score': 'sentiment_score'
                })

                if df_reviews.empty:
                    df_reviews = new_row
                else:
                    df_reviews = pd.concat([df_reviews, new_row], ignore_index=True)
                
                print(f"✅ Procesada reseña {result['id']} de {filename}")
                review_counter += 1

            except Exception as e:
                print(f"❌ Error procesando reseña en {filename}: {e}")
                continue

    if not df_reviews.empty:
        df_reviews.to_csv(output_csv, index=False)
        print(f"\n✅ Total reseñas procesadas: {len(df_reviews)}")
        print(f"💾 DataFrame guardado en: {output_csv}")
    else:
        print("\n⚠️ No se procesaron reseñas. DataFrame vacío.")
    
    print("--- ✅ Paso 1 Finalizado ---")
    return df_reviews

# ------------------------------------------------------------------------------------------------

def get_embeddings(text: str) -> np.ndarray:
    """Genera el embedding para un texto dado."""
    embeddings = EMBEDDING_MODEL.encode([text])[0]
    return embeddings

def step_2_generate_embedding_database(folder_path="tripadvisor_extractions"):
    """
    Etapa 2: Genera embeddings de títulos, descripciones y reseñas, y los indexa en ChromaDB.
    """
    print("\n--- ⚙️ Iniciando Paso 2: Generación de Base de Datos de Embeddings ---")
    tiempo_inicio = datetime.datetime.now()
    
    ids_to_add = []
    embeddings_to_add = []
    metadata_to_add = []
    review_counter = 1

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    for filename in os.listdir(folder_path):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"❌ Error cargando {filename}: {e}")
                continue

        tittle = data.get("title", "No title available.")
        descriptions_list = data.get("all_descriptions", [])
        descriptions = " ".join(descriptions_list)
        reviews = data.get("reviews", [])
        
        main_description = tittle + " " + descriptions
        main_embedding = get_embeddings(main_description)
        
        metadata = {
            "source_file": filename,
            "title": tittle,
            "descriptions": descriptions,
            "type": "place_description" 
        }

        ids_for_mean_vector = []
        embeddings_for_mean_vector = []
    
        ids_to_add.append(str(review_counter))
        embeddings_to_add.append(main_embedding.tolist())
        metadata_to_add.append(metadata)
        review_counter += 1
        
        embeddings_for_mean_vector.append(main_embedding)
        ids_for_mean_vector.append(str(review_counter))



        for review in reviews:
            review_text = review.get("text", "")
            if review_text: 
                review_embedding = get_embeddings(review_text)
                
                ids_to_add.append(str(review_counter))
                embeddings_to_add.append(review_embedding.tolist())
                
                review_metadata = metadata.copy()
                review_metadata["type"] = "review"
                metadata_to_add.append(review_metadata)

                embeddings_for_mean_vector.append(review_embedding)
                ids_for_mean_vector.append(str(review_counter))

                review_counter += 1
        
        if embeddings_for_mean_vector:
            item_vector = np.mean(np.array(embeddings_for_mean_vector), axis=0)
            
            item_vector_metadata = metadata.copy()
            item_vector_metadata["type"] = "item_mean_vector"
            
            ids_to_add.append(str(review_counter))
            embeddings_to_add.append(item_vector.tolist())
            metadata_to_add.append(item_vector_metadata)
            review_counter += 1
            
        print(f"✅ Procesado archivo: {filename}. Embeddings agregados: {len(ids_to_add)}")

    if ids_to_add:
        try:
            collection.add(
                ids=ids_to_add,
                embeddings=embeddings_to_add,
                metadatas=metadata_to_add
            )
            print(f"\n✅ Total de {len(ids_to_add)} embeddings agregados a ChromaDB.")
        except Exception as e:
            print(f"\n❌ Error al agregar a ChromaDB: {e}")

    tiempo_fin = datetime.datetime.now()
    tiempo_total = tiempo_fin - tiempo_inicio
    print(f"Tiempo total de ejecución del embedding: {tiempo_total.total_seconds():.2f} segundos")
    print("--- ✅ Paso 2 Finalizado ---")

# ------------------------------------------------------------------------------------------------

def generar_user_embedding(survey_dict: dict) -> np.ndarray:
    """
    Etapa 3.1: Usa un LLM para transformar la encuesta del usuario en una frase
    semánticamente rica y luego genera un embedding de esa frase.
    """
    json_formateado = json.dumps(survey_dict, indent=4, sort_keys=False)
    
    prompt = f"""En base a la encuesta respondida por un usuario turista interesado en recomendaciones de lugares para visitar en lima, 
    genera una frase corta que resuma sus intereses y preferencias de viaje. 
    Por ejemplo:
    encuesta: "categorias_interesadas": playa , historia , "Modo de viaje": solo, "Presupuesto": medio "Zona preferida": Miraflores
    resultado
    “Quiero lugares históricos, tranquilos, para ir solo, con buena vista al mar”
    IMPORTANTE: no digas el nombre propio de un lugar como 'Lagranja Azul' solo puedes mencionar nombres genericos como 'restaurant' o 'polleria' etc. RESPONDE UNICAMENTE CON LA FRASE, SIN EXPLICAR NADA MÁS.

    Aqui esta la encuesta:
    {json_formateado}
    """
    
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "text"},
    )
    result = completion.choices[0].message.content.strip()
    print("Resumen generado para embedding:", result)
    
    user_embedding = get_embeddings(result)
    return user_embedding

def step_3_recomendar_lugares(user_survey_dict: dict, n_recomendaciones: int = 5) -> List[dict]:
    """
    Etapa 3.2: Genera el embedding del usuario y consulta ChromaDB para obtener
    las recomendaciones más similares.
    """
    print("\n--- ⚙️ Iniciando Paso 3: Sistema de Recomendación ---")
    
    user_embedding = generar_user_embedding(user_survey_dict)
    
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

    resultados = collection.query(
        query_embeddings=[user_embedding.tolist()], 
        n_results=n_recomendaciones * 5, 
        where={"type": "item_mean_vector"} 
    )
    
    recomendaciones = []
    
    unique_titles = set()
    for i in range(len(resultados['metadatas'][0])):
        metadata = resultados['metadatas'][0][i]
        distance = resultados['distances'][0][i]
        title = metadata['title']
        
        if title not in unique_titles:
            lugar = {
                "title": title,
                "descriptions": metadata['descriptions'],
                "source_file": metadata['source_file'],
                "similarity_score": distance
            }
            recomendaciones.append(lugar)
            unique_titles.add(title)

        if len(recomendaciones) >= n_recomendaciones:
            break

    print("--- Paso 3 Finalizado ---")
    return recomendaciones[:n_recomendaciones]


def run_full_pipeline(process_reviews=False, generate_db=False, run_recommendation=False, folder="tripadvisor_extractions"):
    """
    Orquesta todos los pasos del pipeline.
    """
    print("==============================================")
    print("    ✨ INICIO DEL PIPELINE DE RECOMENDACIÓN ✨")
    print("==============================================")
    
    if process_reviews:
        df = step_1_llm_parse_reviews(folder_path=folder)

    if generate_db:
        step_2_generate_embedding_database(folder_path=folder)
    
    if run_recommendation:
        encuesta_ejemplo = {
            "categorias_interesadas": ["cultura", "gastronomía", "caminatas"],
            "modo_viaje": "pareja",
            "presupuesto": "medio",
            "zona_preferida": "Barranco",
            "ambiente": "tranquilo, con buena vista, al aire libre"
        }
        
        recomendaciones = step_3_recomendar_lugares(encuesta_ejemplo, n_recomendaciones=5)
        
        print("\n🏆 **RECOMENDACIONES PARA EL USUARIO** 🏆")
        print("--- Encuesta de Intereses ---")
        print(json.dumps(encuesta_ejemplo, indent=2, ensure_ascii=False))
        print("-" * 40)

        for idx, lugar in enumerate(recomendaciones, start=1):
            print(f"Recomendación {idx}:")
            print(f"Título: {lugar['title']}")
            print(f"Descripción (Fragmento): {lugar['descriptions'][:100]}...")
            print(f"Fuente: {lugar['source_file']}")
            print(f"Similitud (Distancia Coseno): {lugar['similarity_score']:.4f}") 
            print("-" * 40)
    print("FIN DEL PIPELINE DE RECOMENDACIÓN")

