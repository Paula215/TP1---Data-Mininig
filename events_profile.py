import pandas as pd
import numpy as np
import re
import ast
import nltk
from nltk.corpus import stopwords
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)

print("=" * 80)
print("SISTEMA COMPLETO DE MATRIZ VIBE - EVENTOS JOINNUS")
print("Con metadatos estructurados y ponderación temporal")
print("=" * 80)

# ========= CONFIGURACIÓN =========
# Pesos para diferentes componentes del embedding
WEIGHTS = {
    'title': 0.25,          # Título del evento
    'description': 0.35,    # Descripción principal
    'category': 0.15,       # Categoría del evento
    'location': 0.10,       # Ubicación/venue
    'performers': 0.10,     # Artistas/performers
    'metadata': 0.05        # Otros metadatos (fecha cercana, disponibilidad)
}

# Bonus temporal: eventos próximos tienen más relevancia
TEMPORAL_BOOST = True
MAX_DAYS_BOOST = 90  # Días hacia adelante para aplicar boost

print(f"\n⚙️ CONFIGURACIÓN:")
print(f"  - Pesos componentes: {WEIGHTS}")
print(f"  - Boost temporal: {TEMPORAL_BOOST}")
print(f"  - Días de boost: {MAX_DAYS_BOOST}")

# ========= 1. CARGAR DATOS =========
print("\n📂 Cargando datos de eventos...")

# Cargar dataset
eventos = pd.read_csv("data_entrada/joinnus_events.csv")

print(f"  ✓ Eventos cargados: {len(eventos):,}")

# Verificar columnas disponibles
print(f"\n📋 Columnas disponibles:")
for col in eventos.columns:
    non_null = eventos[col].notna().sum()
    print(f"  • {col}: {non_null}/{len(eventos)} ({non_null/len(eventos)*100:.1f}%)")

# ========= 2. ANÁLISIS EXPLORATORIO =========
print("\n📊 ANÁLISIS EXPLORATORIO")
print("-" * 80)

# Categorías
print(f"\n🏷️ Categorías:")
print(f"  ✓ Categorías únicas: {eventos['category'].nunique()}")
print(f"  ✓ Top 10 categorías:")
for cat, count in eventos['category'].value_counts().head(10).items():
    print(f"    • {cat}: {count}")

# Ubicaciones
print(f"\n📍 Ubicaciones:")
print(f"  ✓ Ubicaciones únicas: {eventos['location'].nunique()}")
print(f"  ✓ Top 10 ubicaciones:")
for loc, count in eventos['location'].value_counts().head(10).items():
    loc_str = str(loc)[:50] + '...' if len(str(loc)) > 50 else str(loc)
    print(f"    • {loc_str}: {count}")

# Organizadores
print(f"\n👥 Organizadores:")
print(f"  ✓ Organizadores únicos: {eventos['organizer'].nunique()}")
print(f"  ✓ Eventos con organizador: {eventos['organizer'].notna().sum()} ({eventos['organizer'].notna().sum()/len(eventos)*100:.1f}%)")

# Disponibilidad de tickets
print(f"\n🎫 Disponibilidad de Tickets:")
if 'ticket_availability' in eventos.columns:
    ticket_dist = eventos['ticket_availability'].value_counts()
    for status, count in ticket_dist.items():
        print(f"  • {status}: {count} ({count/len(eventos)*100:.1f}%)")

# ========= 3. PROCESAR FECHAS Y CALCULAR RELEVANCIA TEMPORAL =========
print("\n📅 Procesando fechas y relevancia temporal...")

# Convertir fecha a datetime
eventos['date'] = pd.to_datetime(eventos['date'], errors='coerce')
current_date = datetime.now()

# Calcular días hasta el evento
eventos['days_until_event'] = (eventos['date'] - current_date).dt.days

# Clasificar eventos
eventos['event_status'] = eventos['days_until_event'].apply(
    lambda x: 'Pasado' if x < 0 else ('Próximo' if x <= 30 else 'Futuro')
)

print(f"  ✓ Distribución temporal:")
for status, count in eventos['event_status'].value_counts().items():
    print(f"    • {status}: {count} ({count/len(eventos)*100:.1f}%)")

# Calcular boost temporal (eventos próximos son más relevantes)
def temporal_boost(days_until):
    """
    Boost para eventos próximos:
    - Eventos en 0-30 días: boost 1.5x
    - Eventos en 30-60 días: boost 1.2x
    - Eventos en 60-90 días: boost 1.1x
    - Otros: boost 1.0x
    """
    if pd.isna(days_until):
        return 1.0
    
    if days_until < 0:  # Evento pasado
        return 0.8
    elif days_until <= 30:
        return 1.5
    elif days_until <= 60:
        return 1.2
    elif days_until <= 90:
        return 1.1
    else:
        return 1.0

if TEMPORAL_BOOST:
    eventos['temporal_weight'] = eventos['days_until_event'].apply(temporal_boost)
    print(f"  ✓ Boost temporal promedio: {eventos['temporal_weight'].mean():.3f}")
else:
    eventos['temporal_weight'] = 1.0

# ========= 4. PROCESAR PERFORMERS =========
print("\n🎤 Procesando performers/artistas...")

def parse_performers(performers_str):
    """Parsea lista de performers"""
    if pd.isna(performers_str) or performers_str == '':
        return []
    try:
        if isinstance(performers_str, str):
            # Intentar evaluar como lista
            try:
                performers_list = ast.literal_eval(performers_str)
            except:
                # Si falla, dividir por comas
                performers_list = [p.strip() for p in performers_str.split(',')]
        else:
            performers_list = performers_str
        
        return [p for p in performers_list if p and p.strip()]
    except:
        return []

eventos['performers_list'] = eventos['performer_list'].apply(parse_performers)
eventos['performers_text'] = eventos['performers_list'].apply(lambda x: ' '.join(x))

# Estadísticas de performers
total_performers = sum(len(p) for p in eventos['performers_list'])
eventos_con_performers = (eventos['performers_list'].apply(len) > 0).sum()

print(f"  ✓ Total de performers: {total_performers}")
print(f"  ✓ Eventos con performers: {eventos_con_performers} ({eventos_con_performers/len(eventos)*100:.1f}%)")
print(f"  ✓ Promedio performers/evento: {total_performers/len(eventos):.2f}")

# Top performers
all_performers = []
for performers in eventos['performers_list']:
    all_performers.extend(performers)

from collections import Counter
performer_counts = Counter(all_performers)
if len(performer_counts) > 0:
    print(f"  ✓ Top 10 performers:")
    for performer, count in performer_counts.most_common(10):
        print(f"    • {performer}: {count} eventos")

# ========= 5. LIMPIAR TEXTOS =========
print("\n🧹 Limpiando textos...")

stop_words = set(stopwords.words('spanish')).union(stopwords.words('english'))
# Agregar stopwords personalizadas
stop_words.update(['lima', 'perú', 'peru', 'evento', 'eventos', 'entradas', 'tickets'])

def clean_text(text):
    """Limpia y normaliza texto"""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)        # Caracteres especiales
    text = re.sub(r'\s+', ' ', text).strip()            # Espacios extra
    tokens = [word for word in text.split() if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Limpiar campos principales
print("  [1/6] Limpiando títulos...")
eventos['title_clean'] = eventos['title'].apply(clean_text)

print("  [2/6] Limpiando descripciones...")
eventos['description_clean'] = eventos['description'].apply(clean_text)

print("  [3/6] Limpiando categorías...")
eventos['category_clean'] = eventos['category'].apply(clean_text)

print("  [4/6] Limpiando ubicaciones...")
eventos['location_clean'] = eventos['location'].apply(clean_text)

print("  [5/6] Limpiando performers...")
eventos['performers_clean'] = eventos['performers_text'].apply(clean_text)

print("  [6/6] Limpiando organizadores...")
eventos['organizer_clean'] = eventos['organizer'].apply(clean_text)

print(f"  ✓ Textos limpiados correctamente")

# ========= 6. CREAR TEXTO DE METADATOS =========
print("\n📝 Creando texto de metadatos...")

def metadata_text(row):
    """Crea texto descriptivo de metadatos del evento"""
    parts = []
    
    # Proximidad temporal
    if not pd.isna(row['days_until_event']):
        days = row['days_until_event']
        if days < 0:
            parts.append('evento pasado')
        elif days <= 7:
            parts.append('evento próximo esta semana urgente')
        elif days <= 30:
            parts.append('evento próximo este mes')
        elif days <= 90:
            parts.append('evento próximo trimestre')
        else:
            parts.append('evento futuro')
    
    # Disponibilidad
    if not pd.isna(row['ticket_availability']):
        avail = str(row['ticket_availability']).lower()
        if 'disponible' in avail or 'available' in avail:
            parts.append('entradas disponibles')
        elif 'agotado' in avail or 'sold' in avail:
            parts.append('entradas agotadas alta demanda')
    
    # Tiene organizador reconocido
    if not pd.isna(row['organizer']) and row['organizer'] != '':
        parts.append('organizador verificado')
    
    # Tiene performers
    if len(row['performers_list']) > 0:
        parts.append('con artistas')
        if len(row['performers_list']) > 3:
            parts.append('múltiples artistas')
    
    return ' '.join(parts)

eventos['metadata_text'] = eventos.apply(metadata_text, axis=1)

print(f"  ✓ Metadatos generados")
print(f"  ✓ Ejemplo: '{eventos['metadata_text'].iloc[0]}'")

# ========= 7. CARGAR MODELO DE EMBEDDINGS =========
print("\n🧠 Cargando modelo SentenceTransformer...")

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print(f"  ✓ Modelo cargado: {model.get_sentence_embedding_dimension()} dimensiones")

# ========= 8. GENERAR EMBEDDINGS POR COMPONENTE =========
print("\n✨ Generando embeddings por componente...")

# 8.1 Embeddings de TÍTULO
print("  [1/6] Títulos...")
eventos['title_embedding'] = eventos['title_clean'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

# 8.2 Embeddings de DESCRIPCIÓN
print("  [2/6] Descripciones...")
eventos['description_embedding'] = eventos['description_clean'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

# 8.3 Embeddings de CATEGORÍA
print("  [3/6] Categorías...")
eventos['category_embedding'] = eventos['category_clean'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

# 8.4 Embeddings de UBICACIÓN
print("  [4/6] Ubicaciones...")
eventos['location_embedding'] = eventos['location_clean'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

# 8.5 Embeddings de PERFORMERS
print("  [5/6] Performers...")
eventos['performers_embedding'] = eventos['performers_clean'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

# 8.6 Embeddings de METADATOS
print("  [6/6] Metadatos...")
eventos['metadata_embedding'] = eventos['metadata_text'].apply(
    lambda x: model.encode(x) if len(x) > 0 else np.zeros(384)
)

print(f"  ✓ Embeddings generados para {len(eventos)} eventos")

# ========= 9. COMBINAR EMBEDDINGS CON PESOS =========
print("\n⚡ Creando embedding final combinado...")

def combine_embeddings(row):
    """
    Combina todos los embeddings con pesos configurados
    Aplica boost temporal si está habilitado
    """
    emb_title = row['title_embedding'] * WEIGHTS['title']
    emb_description = row['description_embedding'] * WEIGHTS['description']
    emb_category = row['category_embedding'] * WEIGHTS['category']
    emb_location = row['location_embedding'] * WEIGHTS['location']
    emb_performers = row['performers_embedding'] * WEIGHTS['performers']
    emb_metadata = row['metadata_embedding'] * WEIGHTS['metadata']
    
    # Suma ponderada
    final_emb = (
        emb_title + 
        emb_description + 
        emb_category + 
        emb_location + 
        emb_performers + 
        emb_metadata
    )
    
    # Aplicar boost temporal
    if TEMPORAL_BOOST:
        final_emb = final_emb * row['temporal_weight']
    
    # Normalizar
    norm = np.linalg.norm(final_emb)
    if norm > 0:
        final_emb = final_emb / norm
    
    return final_emb

eventos['V_Evento'] = eventos.apply(combine_embeddings, axis=1)

print(f"  ✓ Embeddings finales creados")

# ========= 10. CREAR TABLA EVENTO_VIBE =========
print("\n📊 Creando tabla EVENTO_VIBE...")

EVENTO_VIBE = eventos[[
    'event_id', 'title', 'category', 'location', 
    'date', 'event_status', 'organizer',
    'V_Evento'
]].copy()

# Convertir embeddings a listas para JSON
EVENTO_VIBE['V_Evento_list'] = EVENTO_VIBE['V_Evento'].apply(
    lambda x: x.tolist() if isinstance(x, np.ndarray) else []
)

# Tabla final minimal
EVENTO_VIBE_FINAL = EVENTO_VIBE[['event_id', 'V_Evento_list']].rename(
    columns={'V_Evento_list': 'V_Evento'}
)

# ========= 11. GUARDAR RESULTADOS =========
print("\n💾 Guardando resultados...")

# Guardar versión completa (con metadatos)
EVENTO_VIBE.to_json('EVENTO_VIBE_complete.json', orient='records', lines=True, force_ascii=False)
print(f"  ✓ Guardado: EVENTO_VIBE_complete.json ({len(EVENTO_VIBE)} eventos)")

# Guardar versión minimal (solo event_id y embedding)
EVENTO_VIBE_FINAL.to_json('EVENTO_VIBE.json', orient='records', lines=True)
print(f"  ✓ Guardado: EVENTO_VIBE.json")

# Guardar también como pickle
import pickle
with open('EVENTO_VIBE.pkl', 'wb') as f:
    pickle.dump(EVENTO_VIBE, f)
print(f"  ✓ Guardado: EVENTO_VIBE.pkl")

# Guardar CSV con metadata (sin embeddings para tamaño)
eventos_metadata = eventos[[
    'event_id', 'title', 'category', 'location', 'date', 
    'event_status', 'days_until_event', 'temporal_weight',
    'ticket_availability', 'organizer'
]].copy()
eventos_metadata.to_csv('EVENTO_VIBE_metadata.csv', index=False)
print(f"  ✓ Guardado: EVENTO_VIBE_metadata.csv")

# ========= 12. ESTADÍSTICAS FINALES =========
print("\n" + "=" * 80)
print("📊 ESTADÍSTICAS FINALES")
print("=" * 80)

print(f"\n🎯 COBERTURA:")
print(f"  - Total de eventos: {len(EVENTO_VIBE):,}")
print(f"  - Eventos próximos (30 días): {(eventos['event_status'] == 'Próximo').sum()}")
print(f"  - Eventos futuros: {(eventos['event_status'] == 'Futuro').sum()}")
print(f"  - Eventos pasados: {(eventos['event_status'] == 'Pasado').sum()}")

print(f"\n📈 DISTRIBUCIÓN POR CATEGORÍA:")
for cat, count in eventos['category'].value_counts().head(10).items():
    print(f"  • {cat}: {count}")

print(f"\n📍 DISTRIBUCIÓN POR UBICACIÓN (Top 10):")
for loc, count in eventos['location'].value_counts().head(10).items():
    loc_str = str(loc)[:40] + '...' if len(str(loc)) > 40 else str(loc)
    print(f"  • {loc_str}: {count}")

print(f"\n⏰ PRÓXIMOS EVENTOS (7 días):")
proximos = eventos[eventos['days_until_event'].between(0, 7)].nsmallest(5, 'days_until_event')
for idx, row in proximos.iterrows():
    print(f"  • {row['title'][:50]}")
    print(f"    Fecha: {row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'}")
    print(f"    Categoría: {row['category']}")
    print(f"    Boost: {row['temporal_weight']:.2f}x")

# ========= 13. VALIDACIÓN DE CALIDAD =========
print(f"\n✅ VALIDACIÓN DE CALIDAD:")

# Verificar dimensionalidad
sample_emb = EVENTO_VIBE['V_Evento'].iloc[0]
print(f"  ✓ Dimensión de embeddings: {len(sample_emb)}")

# Verificar no-zeros
non_zero_count = sum([np.count_nonzero(emb) > 0 for emb in EVENTO_VIBE['V_Evento']])
print(f"  ✓ Embeddings no-vacíos: {non_zero_count}/{len(EVENTO_VIBE)} ({non_zero_count/len(EVENTO_VIBE)*100:.1f}%)")

# Calcular similitud promedio (muestra)
from sklearn.metrics.pairwise import cosine_similarity

sample_size = min(100, len(EVENTO_VIBE))
sample_embeddings = np.array(EVENTO_VIBE['V_Evento'].sample(sample_size).tolist())
sim_matrix = cosine_similarity(sample_embeddings)
np.fill_diagonal(sim_matrix, 0)
avg_similarity = sim_matrix.mean()

print(f"  ✓ Similitud promedio (muestra): {avg_similarity:.3f}")

# ========= 14. EJEMPLOS DE EVENTOS =========
print(f"\n🎭 EJEMPLOS DE EVENTOS PROCESADOS:")
print("-" * 80)

for idx in range(min(5, len(EVENTO_VIBE))):
    row = EVENTO_VIBE.iloc[idx]
    print(f"\n{idx + 1}. {row['title']}")
    print(f"   Categoría: {row['category']}")
    print(f"   Ubicación: {row['location'][:50]}...")
    print(f"   Fecha: {row['date'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['date']) else 'N/A'}")
    print(f"   Status: {row['event_status']}")
    print(f"   Embedding: [{row['V_Evento'][0]:.4f}, {row['V_Evento'][1]:.4f}, ..., {row['V_Evento'][-1]:.4f}]")

# ========= 15. RECOMENDACIONES DE USO =========
print("\n" + "=" * 80)
print("💡 CÓMO USAR LA MATRIZ VIBE DE EVENTOS")
print("=" * 80)
print("""
1. BÚSQUEDA POR SIMILITUD:
   - Encuentra eventos similares por contenido
   - Filtra por categoría, fecha, ubicación

2. RECOMENDACIONES PERSONALIZADAS:
   - Crea perfil de usuario basado en eventos pasados
   - Recomienda eventos con embeddings similares

3. SISTEMA HÍBRIDO:
   - Combina similitud semántica con filtros temporales
   - Boost automático para eventos próximos
   - Penalización para eventos pasados

4. CLUSTERING Y DESCUBRIMIENTO:
   - Agrupa eventos similares
   - Identifica nichos y tendencias
   - Detecta eventos únicos/outliers

5. ACTUALIZACIÓN:
   - Re-generar embeddings semanalmente
   - Ajustar boost temporal dinámicamente

EJEMPLO DE USO:
```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cargar
vibe = pd.read_json('EVENTO_VIBE_complete.json', lines=True)
embeddings = np.array(vibe['V_Evento'].tolist())

# Buscar eventos similares
idx = 0
similarities = cosine_similarity([embeddings[idx]], embeddings)[0]
top_5 = similarities.argsort()[-6:-1][::-1]

print(f"Eventos similares a {vibe.iloc[idx]['title']}:")
for i in top_5:
    print(f"  {vibe.iloc[i]['title']} (sim: {similarities[i]:.3f})")
    print(f"    Fecha: {vibe.iloc[i]['date']}")
```
""")

print("=" * 80)
print("✅ SISTEMA DE MATRIZ VIBE DE EVENTOS COMPLETADO EXITOSAMENTE")
print("=" * 80)