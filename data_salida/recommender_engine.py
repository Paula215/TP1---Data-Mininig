"""
Sistema de recomendaciones basado en embeddings vectoriales
"""
import numpy as np
from pymongo import MongoClient
from typing import List, Tuple, Optional
import os
from dotenv import load_dotenv
from bson import ObjectId
from utils.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

# Configuración de MongoDB
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "recommendations-system")

# Pesos para diferentes tipos de interacción
INTERACTION_WEIGHTS = {
    "view": 0.1,      # Ver un item (menor peso)
    "click": 0.2,     # Click en un item
    "share": 0.3,     # Compartir
    "save": 0.5,      # Guardar para después (peso medio)
    "visit": 0.7,     # Marcar como visitado (peso alto)
    "like": 0.8,      # Like (peso muy alto)
}

# Tasa de aprendizaje (alpha) - controla qué tan rápido se actualiza el perfil del usuario
LEARNING_RATE = 0.15

# Dimensión de los embeddings (debe coincidir con tu modelo)
EMBEDDING_DIM = 384  # Cambiar si usas otro modelo


def update_user_vector(
    user_vec: np.ndarray,
    item_vec: np.ndarray,
    interaction_type: str = "view",
    alpha: float = LEARNING_RATE
) -> np.ndarray:
    """
    Actualiza el vector de embedding del usuario basado en una interacción.
    
    Usa Exponential Moving Average (EMA) para combinar el vector actual del usuario
    con el vector del item con el que interactuó.
    
    Args:
        user_vec: Vector actual del usuario (numpy array)
        item_vec: Vector del item con el que interactuó (numpy array)
        interaction_type: Tipo de interacción ("view", "like", "save", etc.)
        alpha: Tasa de aprendizaje (0-1), controla velocidad de actualización
        
    Returns:
        Nuevo vector del usuario normalizado
    """
    # Obtener peso según tipo de interacción
    weight = INTERACTION_WEIGHTS.get(interaction_type, 0.1)
    
    # Fórmula EMA: new_vec = (1-α) * user_vec + α * weight * item_vec
    new_vec = (1 - alpha) * user_vec + alpha * weight * item_vec
    
    # Normalizar el vector (importante para búsqueda coseno)
    norm = np.linalg.norm(new_vec)
    if norm > 0:
        new_vec = new_vec / norm
    
    return new_vec


def get_item_embedding(
    item_id: str,
    item_type: str,
    connection_string: str = MONGO_URI,
    db_name: str = DB_NAME
) -> Optional[np.ndarray]:
    """
    Obtiene el embedding de un item (SOLO events por ahora, places no disponibles).
    
    Args:
        item_id: ID del item (event_id)
        item_type: "place" o "event"
        connection_string: URI de MongoDB
        db_name: Nombre de la base de datos
        
    Returns:
        Embedding del item como numpy array, o None si no se encuentra
    """
    try:
        client = MongoClient(connection_string)
        db = client[db_name]

        # Ahora los vectores/embeddings están en la colección `combined` y el campo
        # se llama `vector` (Array de dimensión 384).
        collection = db["combined"]

        # Intentar buscar por ObjectId en _id
        query = None
        try:
            oid = ObjectId(item_id)
            query = {"_id": oid}
        except Exception:
            # No es ObjectId: buscar por event_id/place_id si aplica
            if item_type == "event":
                try:
                    query = {"event_id": int(item_id)}
                except Exception:
                    query = {"_id": item_id}
            elif item_type == "place":
                try:
                    query = {"place_id": int(item_id)}
                except Exception:
                    query = {"_id": item_id}
            else:
                query = {"_id": item_id}

        doc = collection.find_one(query, {"vector": 1})

        if doc and "vector" in doc:
            vector = np.array(doc["vector"], dtype=np.float32)
            logger.info("Vector obtenido para %s %s: shape %s", item_type, item_id, vector.shape)
            return vector
        else:
            logger.warning("No se encontró 'vector' en 'combined' para %s %s", item_type, item_id)
            return None

    except Exception as e:
        logger.exception("Error al obtener vector: %s", e)
        return None
    finally:
        if 'client' in locals():
            client.close()


def get_top_similar_items(
    user_embedding: np.ndarray,
    n: int = 10,
    num_candidates: int = 200,
    connection_string: str = MONGO_URI,
    db_name: str = DB_NAME,
    mix_ratio: float = 0.0  # 0% places (no disponibles), 100% events
) -> List[str]:
    """
    Encuentra los N eventos más similares al perfil del usuario usando búsqueda vectorial.
    NOTA: SOLO retorna eventos por ahora (places no tienen embeddings).
    
    Args:
        user_embedding: Vector de embedding del usuario
        n: Número de recomendaciones a retornar
        num_candidates: Número de candidatos para la búsqueda (mayor = mejor calidad)
        connection_string: URI de MongoDB
        db_name: Nombre de la base de datos
        mix_ratio: NO USADO (siempre 100% events)
        
    Returns:
        Lista de IDs (_id de MongoDB) de eventos recomendados
    """
    try:
        client = MongoClient(connection_string)
        db = client[db_name]
        
        recommended_ids = []

        logger.info("Buscando %d items similares en 'combined'...", n)

        combined_col = db["combined"]

        # Buscar por vector en la colección combined. Filtramos por type="event"
        # para mantener el comportamiento anterior que retornaba sólo eventos.
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": user_embedding.tolist(),
                    "path": "vector",
                    "numCandidates": num_candidates,
                    "limit": n,
                    "index": "vector_index",
                    "filter": {"type": "event"}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "type": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        for doc in combined_col.aggregate(pipeline):
            recommended_ids.append(str(doc["_id"]))
            logger.debug("Item %s %s - Score: %.4f", doc.get('type', '?'), str(doc.get('_id')), doc.get('score', 0))

        logger.info("Encontradas %d recomendaciones desde 'combined'", len(recommended_ids))
        return recommended_ids
        
    except Exception as e:
        logger.exception("Error en búsqueda vectorial: %s", e)
        return []
    finally:
        if 'client' in locals():
            client.close()


def initialize_user_vector(
    user_id: str,
    users_collection,
    dim: int = EMBEDDING_DIM
) -> np.ndarray:
    """
    Inicializa el vector de un usuario nuevo con valores aleatorios pequeños.
    
    Args:
        user_id: ID del usuario
        users_collection: Colección de usuarios de MongoDB
        dim: Dimensión del vector
        
    Returns:
        Vector inicializado y normalizado
    """
    # Vector aleatorio pequeño
    user_vec = np.random.randn(dim).astype(np.float32) * 0.01
    
    # Normalizar
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm
    
    # Guardar en la base de datos
    from bson import ObjectId
    users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"embedding": user_vec.tolist()}}
    )
    
    logger.info("Vector inicializado para usuario %s", user_id)
    return user_vec


def get_user_vector(user_id: str, users_collection) -> Optional[np.ndarray]:
    """
    Obtiene el vector de embedding del usuario desde MongoDB.
    Si no existe, lo inicializa.
    
    Args:
        user_id: ID del usuario
        users_collection: Colección de usuarios de MongoDB
        
    Returns:
        Vector del usuario como numpy array
    """
    from bson import ObjectId
    
    user = users_collection.find_one(
        {"_id": ObjectId(user_id)},
        {"embedding": 1}
    )
    
    if user and "embedding" in user:
        return np.array(user["embedding"], dtype=np.float32)
    else:
        # Si no existe, inicializar
        logger.warning("Usuario %s sin embedding, inicializando...", user_id)
        return initialize_user_vector(user_id, users_collection)


def save_user_vector(
    user_id: str,
    user_vec: np.ndarray,
    users_collection
) -> bool:
    """
    Guarda el vector actualizado del usuario en MongoDB.
    
    Args:
        user_id: ID del usuario
        user_vec: Vector actualizado
        users_collection: Colección de usuarios de MongoDB
        
    Returns:
        True si se guardó correctamente
    """
    from bson import ObjectId
    
    try:
        result = users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"embedding": user_vec.tolist()}}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.exception("Error al guardar vector: %s", e)
        return False


def update_user_recommendations(
    user_id: str,
    interaction_type: str,
    item_id: str,
    item_type: str,
    users_collection,
    n_recommendations: int = 20
) -> dict:
    """
    Función principal: Actualiza el perfil del usuario y genera nuevas recomendaciones.
    
    ADAPTADO: Solo funciona con eventos por ahora (places sin embeddings).
    Si el usuario interactúa con un place, solo actualiza el registro pero no afecta
    las recomendaciones hasta que tengamos embeddings de places.
    
    Args:
        user_id: ID del usuario
        interaction_type: Tipo de interacción ("like", "save", etc.)
        item_id: ID del item (event_id o place_id)
        item_type: "place" o "event"
        users_collection: Colección de usuarios de MongoDB
        n_recommendations: Número de recomendaciones a generar
        
    Returns:
        Dict con información del proceso
    """
    try:
        logger.info("ACTUALIZANDO RECOMENDACIONES | Usuario: %s | Interacción: %s | Item: %s %s", user_id, interaction_type, item_type, item_id)
        
        # 1. Obtener vector actual del usuario
        user_vec = get_user_vector(user_id, users_collection)
        if user_vec is None:
            return {"success": False, "error": "No se pudo obtener vector del usuario"}
        
        # 2. Obtener vector del item
        item_vec = get_item_embedding(item_id, item_type)
        
        # Si es un place sin embedding, usar vector aleatorio pequeño
        if item_vec is None:
            if item_type == "place":
                logger.warning("Place sin embedding, usando actualización genérica")
                # Actualizar muy levemente con ruido aleatorio
                item_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32) * 0.01
                norm = np.linalg.norm(item_vec)
                if norm > 0:
                    item_vec = item_vec / norm
            else:
                logger.warning("No se pudo obtener embedding del evento, usando vector anterior")
                item_vec = user_vec  # Fallback
        
        # 3. Actualizar vector del usuario
        new_user_vec = update_user_vector(
            user_vec,
            item_vec,
            interaction_type=interaction_type
        )
        
        # 4. Buscar SOLO eventos similares
        recommended_ids = get_top_similar_items(
            new_user_vec,
            n=n_recommendations
        )
        
        if not recommended_ids:
            logger.warning("No se encontraron recomendaciones, usando fallback")
            # Obtener eventos populares como fallback
            from pymongo import MongoClient
            import os
            client = MongoClient(os.getenv("MONGO_URI"))
            db = client[os.getenv("DB_NAME")]
            combined = db["combined"]
            
            # Obtener eventos aleatorios
            fallback_events = list(
                combined.find(
                    {"type": "event"},
                    {"_id": 1}
                ).limit(n_recommendations)
            )
            recommended_ids = [str(e["_id"]) for e in fallback_events]
            client.close()
        
        # 5. Guardar en base de datos
        save_user_vector(user_id, new_user_vec, users_collection)
        
        from bson import ObjectId
        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"recommendations": recommended_ids}}
        )
        
        logger.info("PROCESO COMPLETADO | Nuevas recomendaciones: %d", len(recommended_ids))
        
        return {
            "success": True,
            "updated_vector": True,
            "recommended_ids": recommended_ids,
            "num_recommendations": len(recommended_ids)
        }
        
    except Exception as e:
        print(f"❌ Error en update_user_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}