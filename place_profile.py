import pandas as pd
import numpy as np
import re
import ast
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)

print("=" * 80)
print("SISTEMA COMPLETO DE MATRIZ VIBE - GOOGLE PLACES")
print("Con metadatos estructurados (name, address, rating, categoria, types)")
print("=" * 80)

# ========= CONFIGURACIÓN =========
DECAY_YEARS = 5
MIN_WEIGHT = 0.1
RATING_WEIGHT = True

# Pesos para diferentes componentes del embedding
WEIGHTS = {
    'reviews': 0.40,        # Reviews ponderadas temporalmente
    'name': 0.15,           # Nombre del lugar
    'categoria': 0.20,      # Categoría principal
    'types': 0.15,          # Tipos de Google Maps
    'metadata': 0.10        # Rating, ubicación, etc.
}

print(f"\n⚙️ CONFIGURACIÓN:")
print(f"  - Decaimiento temporal: {DECAY_YEARS} años")
print(f"  - Pesos componentes: {WEIGHTS}")

# ========= 1. CARGAR DATOS =========
print("\n📂 Cargando datos...")

# Cargar datos principales
df_places = pd.read_csv('data_entrada/google_places.csv')
df_reviews = pd.read_csv('data_entrada/google_reviews.csv')

print(f"  ✓ Lugares: {len(df_places):,}")
print(f"  ✓ Reviews: {len(df_reviews):,}")

# Verificar columnas disponibles
print(f"\n📋 Columnas en df_places:")
print(f"  {', '.join(df_places.columns.tolist())}")

# ========= 2. PROCESAR TYPES (lista de categorías de Google) =========
print("\n🏷️ Procesando types...")

def parse_types(types_str):
    """Convierte string de lista a lista real y limpia"""
    if pd.isna(types_str):
        return []
    try:
        # Si es string, evaluar como lista
        if isinstance(types_str, str):
            types_list = ast.literal_eval(types_str)
        else:
            types_list = types_str
        
        # Limpiar y filtrar
        types_clean = [
            t.replace('_', ' ').replace('point of interest', '').replace('establishment', '')
            for t in types_list 
            if t not in ['point_of_interest', 'establishment']
        ]
        return [t.strip() for t in types_clean if t.strip()]
    except:
        return []

df_places['types_list'] = df_places['types'].apply(parse_types)
df_places['types_text'] = df_places['types_list'].apply(lambda x: ' '.join(x))

# Análisis de types
all_types = []
for types_list in df_places['types_list']:
    all_types.extend(types_list)

from collections import Counter
type_counts = Counter(all_types)
print(f"  ✓ Types únicos: {len(type_counts)}")
print(f"  ✓ Top 10 types:")
for type_name, count in type_counts.most_common(10):
    print(f"    • {type_name}: {count}")

# ========= 3. EXTRAER DISTRITO DE ADDRESS =========
print("\n📍 Extrayendo información geográfica...")

def extract_distrito(address):
    """Extrae el distrito del address"""
    if pd.isna(address):
        return ''
    
    # Distritos de Lima
    distritos = [
        'miraflores', 'san isidro', 'barranco', 'surco', 'santiago de surco',
        'la molina', 'san borja', 'jesús maría', 'lince', 'magdalena',
        'pueblo libre', 'san miguel', 'callao', 'lima', 'cercado de lima',
        'breña', 'chorrillos', 'surquillo', 'ate', 'santa anita', 'el agustino',
        'san juan de lurigancho', 'los olivos', 'independencia', 'comas',
        'carabayllo', 'puente piedra', 'san martín de porres', 'rímac',
        'villa el salvador', 'villa maría del triunfo', 'san juan de miraflores',
        'la victoria', 'pachacamac', 'lurín', 'pucusana', 'punta hermosa',
        'punta negra', 'san bartolo', 'santa maría del mar', 'santa rosa'
    ]
    
    address_lower = address.lower()
    for distrito in distritos:
        if distrito in address_lower:
            return distrito.title()
    
    return 'Lima'

df_places['distrito'] = df_places['address'].apply(extract_distrito)

print(f"  ✓ Distritos identificados: {df_places['distrito'].nunique()}")
print(f"  ✓ Top 5 distritos:")
for distrito, count in df_places['distrito'].value_counts().head(5).items():
    print(f"    • {distrito}: {count}")

# ========= 4. PROCESAR REVIEWS CON PONDERACIÓN TEMPORAL =========
print("\n⏰ Procesando reviews con ponderación temporal...")

# Función para parsear tiempo relativo tipo "X months ago", "X years ago"
def parse_relative_time(time_str):
    """
    Convierte texto relativo a años
    Ejemplos: 
    - "2 months ago" → 0.167
    - "1 year ago" → 1.0
    - "hace 3 meses" → 0.25
    - "a month ago" → 0.083
    """
    if pd.isna(time_str) or time_str == '':
        return 0.0
    
    time_str = str(time_str).lower().strip()
    
    # Extraer número (si existe)
    import re
    numbers = re.findall(r'\d+', time_str)
    
    # Si dice "a month ago" o "an hour ago", es 1
    if not numbers:
        if any(word in time_str for word in ['a ', 'an ']):
            num = 1
        else:
            return 0.0
    else:
        num = int(numbers[0])
    
    # Determinar unidad (inglés y español)
    if any(word in time_str for word in ['year', 'año', 'años']):
        return num
    elif any(word in time_str for word in ['month', 'mes', 'meses']):
        return num / 12.0
    elif any(word in time_str for word in ['week', 'semana', 'semanas']):
        return num / 52.0
    elif any(word in time_str for word in ['day', 'día', 'dias', 'días']):
        return num / 365.0
    elif any(word in time_str for word in ['hour', 'hora', 'horas']):
        return num / (365.0 * 24)
    elif any(word in time_str for word in ['minute', 'minuto', 'minutos']):
        return num / (365.0 * 24 * 60)
    else:
        return 0.0

# Aplicar parseo
df_reviews['years_old'] = df_reviews['time'].apply(parse_relative_time)

print(f"  ✓ Tiempo parseado correctamente")
print(f"    - Review más antigua: {df_reviews['years_old'].max():.1f} años")
print(f"    - Review más reciente: {df_reviews['years_old'].min():.1f} años")
print(f"    - Promedio de antigüedad: {df_reviews['years_old'].mean():.1f} años")

# Distribución temporal
print(f"\n  📊 Distribución temporal de reviews:")
recent = (df_reviews['years_old'] < 1).sum()
medium = ((df_reviews['years_old'] >= 1) & (df_reviews['years_old'] < 3)).sum()
old = (df_reviews['years_old'] >= 3).sum()
print(f"    • < 1 año: {recent:,} ({recent/len(df_reviews)*100:.1f}%)")
print(f"    • 1-3 años: {medium:,} ({medium/len(df_reviews)*100:.1f}%)")
print(f"    • > 3 años: {old:,} ({old/len(df_reviews)*100:.1f}%)")

# Función de decaimiento temporal
def temporal_weight(years_old, decay_years=DECAY_YEARS, min_weight=MIN_WEIGHT):
    if pd.isna(years_old) or years_old < 0:
        return 1.0
    lambda_decay = np.log(2) / decay_years
    weight = np.exp(-lambda_decay * years_old)
    return max(weight, min_weight)

df_reviews['temporal_weight'] = df_reviews['years_old'].apply(temporal_weight)

# Peso por rating (usar rating_individual)
if RATING_WEIGHT and 'rating_individual' in df_reviews.columns:
    df_reviews['rating_weight'] = df_reviews['rating_individual'].apply(
        lambda r: 0.5 + (r - 1) / 4 if not pd.isna(r) else 1.0
    )
else:
    df_reviews['rating_weight'] = 1.0

# Peso por longitud
df_reviews['text_length'] = df_reviews['review_text'].fillna('').apply(len)

def length_weight(length):
    if length < 50:
        return 0.5
    elif length < 100:
        return 0.7 + (length - 50) / 50 * 0.3
    elif length <= 500:
        return 1.0
    elif length <= 1000:
        return 1.0 - (length - 500) / 500 * 0.2
    else:
        return 0.8

df_reviews['length_weight'] = df_reviews['text_length'].apply(length_weight)

# Peso final
df_reviews['final_weight'] = (
    df_reviews['temporal_weight'] * 
    df_reviews['rating_weight'] * 
    df_reviews['length_weight']
)

df_reviews['norm_weight'] = df_reviews.groupby('place_id')['final_weight'].transform(
    lambda x: x / x.sum() if x.sum() > 0 else 1
)

print(f"\n  ✓ Pesos calculados:")
print(f"    - Peso temporal promedio: {df_reviews['temporal_weight'].mean():.3f}")
print(f"    - Peso por rating promedio: {df_reviews['rating_weight'].mean():.3f}")
print(f"    - Peso por longitud promedio: {df_reviews['length_weight'].mean():.3f}")
print(f"    - Peso final promedio: {df_reviews['final_weight'].mean():.3f}")

# ========= 5. LIMPIAR TEXTOS =========
print("\n🧹 Limpiando textos...")

stop_words = set(stopwords.words('spanish')).union(stopwords.words('english'))
# Agregar stopwords personalizadas
stop_words.update(['lima', 'perú', 'peru', 'lugar', 'sitio', 'local'])

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Limpiar reviews
df_reviews['clean_text'] = df_reviews['review_text'].apply(clean_text)

# Limpiar campos de lugares
df_places['name_clean'] = df_places['name'].apply(clean_text)
df_places['categoria_clean'] = df_places['categoria'].apply(clean_text)
df_places['types_clean'] = df_places['types_text'].apply(clean_text)

print(f"  ✓ Textos limpiados")

# ========= 6. CARGAR MODELO DE EMBEDDINGS =========
print("\n🧠 Cargando modelo SentenceTransformer...")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(f"  ✓ Modelo cargado: {model.get_sentence_embedding_dimension()} dimensiones")

# ========= 7. GENERAR EMBEDDINGS POR COMPONENTE =========
print("\n✨ Generando embeddings por componente...")

# 7.1 Embeddings de REVIEWS ponderadas
print("  [1/5] Reviews ponderadas...")

def weighted_review_embedding(group):
    """Promedio ponderado de embeddings de reviews"""
    if len(group) == 0:
        return np.zeros(384)
    
    embeddings = []
    weights = []
    
    for idx, row in group.iterrows():
        if len(row['clean_text']) > 0:
            emb = model.encode(row['clean_text'])
            embeddings.append(emb)
            weights.append(row['norm_weight'])
    
    if len(embeddings) == 0:
        return np.zeros(384)
    
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    weights_expanded = weights[:, np.newaxis]
    
    weighted_emb = (embeddings * weights_expanded).sum(axis=0)
    return weighted_emb

review_embeddings = (
    df_reviews.groupby('place_id')
    .apply(weighted_review_embedding)
    .reset_index()
    .rename(columns={0: 'review_embedding'})
)

# 7.2 Embeddings de NOMBRE
print("  [2/5] Nombres de lugares...")
df_places['name_embedding'] = df_places['name_clean'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

# 7.3 Embeddings de CATEGORÍA
print("  [3/5] Categorías...")
df_places['categoria_embedding'] = df_places['categoria_clean'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

# 7.4 Embeddings de TYPES
print("  [4/5] Types de Google Maps...")
df_places['types_embedding'] = df_places['types_clean'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

# 7.5 Embeddings de METADATOS (rating + distrito + ubicación)
print("  [5/5] Metadatos (rating + distrito + ubicación)...")

def metadata_text(row):
    """Crea texto descriptivo de metadatos"""
    parts = []
    
    # Rating como texto descriptivo
    if not pd.isna(row['rating']):
        rating = row['rating']
        if rating >= 4.5:
            parts.append('excelente calidad valoración muy alta')
        elif rating >= 4.0:
            parts.append('buena calidad valoración positiva')
        elif rating >= 3.5:
            parts.append('calidad aceptable valoración promedio')
        elif rating >= 3.0:
            parts.append('calidad regular')
        else:
            parts.append('valoración baja necesita mejorar')
    
    # Distrito (ya viene limpio)
    if not pd.isna(row['distrito']) and row['distrito'] != '':
        distrito_clean = str(row['distrito']).lower()
        parts.append(f'ubicado {distrito_clean}')
        parts.append(distrito_clean)  # Repetir para dar más peso
    
    # Categoría (adicional para reforzar)
    if not pd.isna(row['categoria']) and row['categoria'] != '':
        parts.append(str(row['categoria']).lower())
    
    return ' '.join(parts)

df_places['metadata_text'] = df_places.apply(metadata_text, axis=1)
df_places['metadata_embedding'] = df_places['metadata_text'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

print(f"  ✓ Embeddings generados para {len(df_places)} lugares")
print(f"  ✓ Ejemplo de metadata_text: '{df_places['metadata_text'].iloc[0]}'")

# ========= 8. COMBINAR CON REVIEWS =========
print("\n🔗 Combinando embeddings...")

df_merged = pd.merge(
    df_places[['place_id', 'name', 'categoria', 'rating', 'distrito',
               'name_embedding', 'categoria_embedding', 'types_embedding', 
               'metadata_embedding']],
    review_embeddings,
    on='place_id',
    how='left'
)

# Rellenar reviews faltantes con zeros
df_merged['review_embedding'] = df_merged['review_embedding'].apply(
    lambda x: x if isinstance(x, np.ndarray) else np.zeros(384)
)

# ========= 9. CREAR EMBEDDING FINAL COMBINADO =========
print("\n⚡ Creando embedding final combinado...")

def combine_embeddings(row):
    """
    Combina todos los embeddings con pesos configurados
    """
    emb_reviews = row['review_embedding'] * WEIGHTS['reviews']
    emb_name = row['name_embedding'] * WEIGHTS['name']
    emb_categoria = row['categoria_embedding'] * WEIGHTS['categoria']
    emb_types = row['types_embedding'] * WEIGHTS['types']
    emb_metadata = row['metadata_embedding'] * WEIGHTS['metadata']
    
    # Suma ponderada
    final_emb = emb_reviews + emb_name + emb_categoria + emb_types + emb_metadata
    
    # Normalizar (opcional pero recomendado)
    norm = np.linalg.norm(final_emb)
    if norm > 0:
        final_emb = final_emb / norm
    
    return final_emb

df_merged['V_Lugar'] = df_merged.apply(combine_embeddings, axis=1)

print(f"  ✓ Embeddings finales creados")

# ========= 10. CREAR TABLA LUGAR_VIBE =========
print("\n📊 Creando tabla LUGAR_VIBE...")

LUGAR_VIBE = df_merged[['place_id', 'name', 'categoria', 'rating', 'distrito', 'V_Lugar']].copy()

# Convertir embeddings a listas para JSON
LUGAR_VIBE['V_Lugar_list'] = LUGAR_VIBE['V_Lugar'].apply(
    lambda x: x.tolist() if isinstance(x, np.ndarray) else []
)

# Tabla final
LUGAR_VIBE_FINAL = LUGAR_VIBE[['place_id', 'V_Lugar_list']].rename(columns={'V_Lugar_list': 'V_Lugar'})

# ========= 11. GUARDAR RESULTADOS =========
print("\n💾 Guardando resultados...")

# Guardar versión completa (con metadatos)
LUGAR_VIBE.to_json('LUGAR_VIBE_complete.json', orient='records', lines=True, force_ascii=False)
print(f"  ✓ Guardado: LUGAR_VIBE_complete.json ({len(LUGAR_VIBE)} lugares)")

# Guardar versión minimal (solo place_id y embedding)
LUGAR_VIBE_FINAL.to_json('LUGAR_VIBE.json', orient='records', lines=True)
print(f"  ✓ Guardado: LUGAR_VIBE.json")

# Guardar también como pickle para uso en Python
import pickle
with open('LUGAR_VIBE.pkl', 'wb') as f:
    pickle.dump(LUGAR_VIBE, f)
print(f"  ✓ Guardado: LUGAR_VIBE.pkl")

# ========= 12. ESTADÍSTICAS Y ANÁLISIS =========
print("\n" + "=" * 80)
print("📊 ESTADÍSTICAS FINALES")
print("=" * 80)

print(f"\n🎯 COBERTURA:")
print(f"  - Total de lugares: {len(LUGAR_VIBE):,}")
print(f"  - Lugares con reviews: {len(review_embeddings):,}")
print(f"  - Lugares sin reviews: {len(LUGAR_VIBE) - len(review_embeddings):,}")

print(f"\n📈 DISTRIBUCIÓN POR CATEGORÍA:")
top_cats = LUGAR_VIBE['categoria'].value_counts().head(10)
for cat, count in top_cats.items():
    print(f"  • {cat}: {count}")

print(f"\n📍 DISTRIBUCIÓN POR DISTRITO:")
top_dist = LUGAR_VIBE['distrito'].value_counts().head(10)
for dist, count in top_dist.items():
    print(f"  • {dist}: {count}")

print(f"\n⭐ DISTRIBUCIÓN DE RATINGS:")
rating_bins = pd.cut(LUGAR_VIBE['rating'], bins=[0, 3, 3.5, 4, 4.5, 5], 
                     labels=['<3', '3-3.5', '3.5-4', '4-4.5', '4.5-5'])
print(rating_bins.value_counts().sort_index())

# ========= 13. VALIDACIÓN DE CALIDAD =========
print(f"\n✅ VALIDACIÓN DE CALIDAD:")

# Verificar dimensionalidad
sample_emb = LUGAR_VIBE['V_Lugar'].iloc[0]
print(f"  ✓ Dimensión de embeddings: {len(sample_emb)}")

# Verificar no-zeros
non_zero_count = sum([np.count_nonzero(emb) > 0 for emb in LUGAR_VIBE['V_Lugar']])
print(f"  ✓ Embeddings no-vacíos: {non_zero_count}/{len(LUGAR_VIBE)} ({non_zero_count/len(LUGAR_VIBE)*100:.1f}%)")

# Calcular similitud promedio (muestra)
from sklearn.metrics.pairwise import cosine_similarity

sample_size = min(100, len(LUGAR_VIBE))
sample_embeddings = np.array(LUGAR_VIBE['V_Lugar'].sample(sample_size).tolist())
sim_matrix = cosine_similarity(sample_embeddings)

# Eliminar diagonal (similitud consigo mismo)
np.fill_diagonal(sim_matrix, 0)
avg_similarity = sim_matrix.mean()

print(f"  ✓ Similitud promedio (muestra): {avg_similarity:.3f}")
print(f"    (Valores entre 0.2-0.4 indican buena diferenciación)")

# ========= 14. EJEMPLOS DE LUGARES =========
print(f"\n🏛️ EJEMPLOS DE LUGARES PROCESADOS:")
print("-" * 80)

for idx in range(min(5, len(LUGAR_VIBE))):
    row = LUGAR_VIBE.iloc[idx]
    print(f"\n{idx + 1}. {row['name']}")
    print(f"   Categoría: {row['categoria']}")
    print(f"   Rating: {row['rating']:.1f}⭐")
    print(f"   Distrito: {row['distrito']}")
    print(f"   Embedding: [{row['V_Lugar'][0]:.4f}, {row['V_Lugar'][1]:.4f}, ..., {row['V_Lugar'][-1]:.4f}]")

# ========= 15. RECOMENDACIONES DE USO =========
print("\n" + "=" * 80)
print("💡 CÓMO USAR LA MATRIZ VIBE")
print("=" * 80)
print("""
1. BÚSQUEDA POR SIMILITUD:
   - Usa cosine_similarity para encontrar lugares similares
   - Filtra por categoría/distrito para búsquedas específicas

2. RECOMENDACIONES PERSONALIZADAS:
   - Crea perfil de usuario como promedio de lugares visitados
   - Busca lugares con alta similitud al perfil

3. CLUSTERING:
   - Agrupa lugares similares (K-means, DBSCAN)
   - Identifica "zonas vibe" en la ciudad

4. BÚSQUEDA HÍBRIDA:
   - Combina similitud de embeddings con filtros (precio, rating, distrito)
   - Sistema de scoring multi-criterio

5. ACTUALIZACIÓN PERIÓDICA:
   - Re-generar embeddings mensualmente
   - Incluir nuevas reviews con mayor peso temporal

EJEMPLO DE USO:
```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar
vibe = pd.read_json('LUGAR_VIBE_complete.json', lines=True)
embeddings = np.array(vibe['V_Lugar'].tolist())

# Buscar similares a un lugar
idx = 0
similarities = cosine_similarity([embeddings[idx]], embeddings)[0]
top_5 = similarities.argsort()[-6:-1][::-1]

print(f"Lugares similares a {vibe.iloc[idx]['name']}:")
for i in top_5:
    print(f"  {vibe.iloc[i]['name']} (sim: {similarities[i]:.3f})")
```
""")

print("=" * 80)
print("✅ SISTEMA DE MATRIZ VIBE COMPLETADO EXITOSAMENTE")
print("=" * 80)