"""
Collaborative Filtering and Recommendation System
==================================================

This module implements a collaborative filtering recommendation system that:
1. Builds user preference vectors from their interactions (likes, saves, visits)
2. Finds similar users based on vector similarity
3. Generates recommendations using collaborative filtering
4. Supports hybrid recommendations combining CF and content-based approaches

Main Components:
- User vector calculation from interactions
- Similarity search for users and events
- Collaborative filtering recommendations
- Hybrid recommendation system
"""

# ============================================================================
# PART 1: IMPORTS AND DEPENDENCIES
# ============================================================================

from collections import Counter
import datetime
import os
import sys
from pymongo import MongoClient
from bson import ObjectId
import numpy as np
import math


# ============================================================================
# PART 2: CONFIGURATION CONSTANTS
# ============================================================================

# Interaction strength weights
# Higher values mean the interaction type has more influence on user preferences
LIKE_STRENGTH = 1.0   # Weight for "like" interactions
SAVE_STRENGTH = 2.0   # Weight for "save" interactions (stronger signal)
VISIT_STRENGTH = 0.5  # Weight for "visit" interactions (weaker signal)

# Time decay parameter for recency weighting
# Higher values mean older interactions have less weight
# 0.0 = no time decay (all interactions weighted equally)
LAMBDA_DECAY = 0.01


# ============================================================================
# PART 3: DATABASE CONNECTION
# ============================================================================

# MongoDB connection string
uri = "mongodb+srv://mrclpgg_db_user:K9NMlwFHZpeltCwI@cluster0.qdopesi.mongodb.net/?appName=Cluster0"

# Initialize MongoDB client and database references
client = MongoClient(uri)
user_db = client.get_database("turislima_db").users      # User collection
data_db = client.get_database("turislima_db").combined   # Events collection


# ============================================================================
# PART 4: SIMILARITY CALCULATION UTILITIES
# ============================================================================

def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical). Values near 1 indicate
    high similarity.
    
    Args:
        a: First vector (list or numpy array)
        b: Second vector (list or numpy array)
    
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dot_similarity(a, b):
    """
    Calculate dot product similarity between two vectors.
    
    Dot product is used for normalized vectors (where cosine similarity
    equals dot product). Useful when vectors are already normalized.
    
    Args:
        a: First vector (list or numpy array)
        b: Second vector (list or numpy array)
    
    Returns:
        float: Dot product of the two vectors
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.dot(a, b))


# ============================================================================
# PART 5: USER INTERACTION DATA RETRIEVAL
# ============================================================================

def get_all_user_interactions(user_id):
    """
    Retrieve all user interactions (likes, saves, visits) with event vectors.
    
    Fetches the user's interaction history and enriches each interaction
    with the corresponding event's embedding vector for vector calculations.
    
    Args:
        user_id: ObjectId of the user
    
    Returns:
        tuple: (likes, saves, visits) where each is a list of dicts containing:
               - 'id': event ObjectId
               - 'ts': timestamp
               - 'vector': event embedding vector
    
    Raises:
        ValueError: If user is not found
    """
    # Fetch user document from database
    user = user_db.find_one({'_id': user_id})
    if user is None:
        raise ValueError(f"user not found: {user_id}")

    # Extract interaction lists (default to empty lists if not present)
    likes = user.get('likes', [])
    saves = user.get('saves', [])
    visits = user.get('visits', [])

    def _fetch_embeddings(actions: list[dict]):
        """
        Helper function to fetch event vectors for a list of interactions.
        
        Args:
            actions: List of interaction dicts with 'id' field
        
        Returns:
            List of interaction dicts with added 'vector' field
        """
        if not actions:
            return []
        
        # Extract event IDs from interactions
        ids = [ObjectId(action['id']) for action in actions]
        
        # Fetch event documents with their vectors
        docs = data_db.find({'_id': {'$in': ids}}, {'vector': 1})
        
        # Create mapping from event ID to vector
        id_to_emb = {doc['_id']: doc.get('vector') for doc in docs}
        
        # Enrich interactions with their corresponding vectors
        return [{**action, 'vector': id_to_emb[ObjectId(action['id'])]} 
                for action in actions]

    # Fetch vectors for each interaction type
    likes = _fetch_embeddings(likes)
    saves = _fetch_embeddings(saves)
    visits = _fetch_embeddings(visits)

    return likes, saves, visits


def get_user_seen_event_ids(user_id):
    """
    Get set of all event IDs that the user has interacted with.
    
    Used to exclude already-interacted events from recommendations.
    
    Args:
        user_id: ObjectId of the user
    
    Returns:
        set: Set of ObjectIds representing events the user has interacted with
    """
    user = user_db.find_one({'_id': user_id}, {'seen': 1})
    return user.get('seen', [])


def get_events_from_user(user_id, interaction_types=None):
    """
    Get all events that a user has interacted with, with their weights.
    
    Args:
        user_id: ObjectId of the user
        interaction_types: List of interaction types to include 
                           ['likes', 'saves', 'visits']
                           If None, includes all types
    
    Returns:
        List of tuples: (event_id, interaction_type, weight)
    """
    if interaction_types is None:
        interaction_types = ['likes', 'saves', 'visits']
    
    user = user_db.find_one({'_id': user_id}, {'likes': 1, 'saves': 1, 'visits': 1})
    if user is None:
        return []
    
    # Map interaction types to their strength weights
    interaction_weights = {
        'likes': LIKE_STRENGTH,
        'saves': SAVE_STRENGTH,
        'visits': VISIT_STRENGTH
    }
    
    events = []
    for interaction_type in interaction_types:
        weight = interaction_weights.get(interaction_type, 1.0)
        for interaction in user.get(interaction_type, []):
            event_id = ObjectId(interaction['id'])
            events.append((event_id, interaction_type, weight))
    
    return events


# ============================================================================
# PART 6: USER VECTOR CALCULATION (FULL RECALCULATION)
# ============================================================================

def compute_interaction_weight(ts, strength, lambda_decay=LAMBDA_DECAY, now=None):
    """
    Calculate the weight of an interaction based on time decay.
    
    Uses exponential decay: weight = strength * exp(-lambda * age_in_days)
    Older interactions get lower weights if lambda_decay > 0.
    
    Args:
        ts: Timestamp of the interaction (datetime)
        strength: Base strength of the interaction type
        lambda_decay: Decay rate (0 = no decay)
        now: Current time (defaults to UTC now)
    
    Returns:
        float: Weighted interaction strength
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    
    # Ensure timestamp is timezone-aware
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
    
    # Calculate age in days
    age_days = (now - ts).total_seconds() / 86400.0
    
    # Apply exponential decay
    decay = math.exp(-lambda_decay * age_days)
    
    return strength * decay


def get_full_recalc_user_vector(user_id, now=None, lambda_decay=0.05):
    """
    Recalculate user preference vector from all interactions.
    
    This function:
    1. Fetches all user interactions (likes, saves, visits)
    2. Weights each interaction by type and recency
    3. Computes weighted average of event vectors
    4. Normalizes the result
    
    Args:
        user_id: ObjectId of the user
        now: Current time for decay calculation (defaults to UTC now)
        lambda_decay: Time decay parameter (overrides global LAMBDA_DECAY)
    
    Returns:
        tuple: (normalized_vector, total_weight)
               - normalized_vector: numpy array representing user preferences
               - total_weight: Sum of all interaction weights
    """
    # Get all user interactions with their event vectors
    likes, saves, visits = get_all_user_interactions(user_id)

    # Use current UTC time if not provided
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)

    total_weight = 0.0
    vector_sum = None  # Will be initialized as numpy array on first use

    def add_items(items, strength):
        """
        Add weighted event vectors to the running sum.
        
        Args:
            items: List of interactions with 'ts' and 'vector' fields
            strength: Base strength for this interaction type
        """
        nonlocal vector_sum, total_weight

        for item in items:
            ts = item["ts"]
            
            # Ensure timestamp is timezone-aware (UTC)
            if isinstance(ts, datetime.datetime):
                if ts.tzinfo is None:
                    # Make naive datetime timezone-aware (assume UTC)
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
            
            # Convert event vector to numpy array
            vec = np.array(item["vector"], dtype=float)

            # Calculate age in days
            age_days = (now - ts).total_seconds() / 86400.0
            
            # Apply exponential decay
            decay = math.exp(-lambda_decay * age_days)

            # Calculate final weight: strength * decay
            weight = strength * decay
            
            # Weight the vector
            weighted_vec = vec * weight

            # Accumulate weighted vectors
            if vector_sum is None:
                vector_sum = weighted_vec
            else:
                vector_sum += weighted_vec

            # Accumulate total weight
            total_weight += weight

    # Process all interaction types with their respective strengths
    add_items(likes, strength=LIKE_STRENGTH)
    add_items(saves, strength=SAVE_STRENGTH)
    add_items(visits, strength=VISIT_STRENGTH)

    # If user has no interactions, return None
    if vector_sum is None:
        return None, 0.0

    # Normalize the vector (unit vector)
    norm = np.linalg.norm(vector_sum)
    if norm == 0:
        normalized = vector_sum
    else:
        normalized = vector_sum / norm

    return normalized, total_weight


def recalc_user_vector(user_id, now=None, lambda_decay=0.05):
    """
    Recalculate and update user vector in the database.
    
    This is the main function to call when you want to update a user's
    preference vector after they've had new interactions.
    
    Args:
        user_id: ObjectId of the user
        now: Current time for decay calculation
        lambda_decay: Time decay parameter
    
    Raises:
        ValueError: If user is not found
    """
    # Calculate new vector
    vector, total_weight = get_full_recalc_user_vector(user_id, now, lambda_decay)
    
    # Verify user exists
    user = user_db.find_one({'_id': user_id})
    if user is None:
        raise ValueError(f"user not found: {user_id}")
    
    # Convert numpy array to list for MongoDB storage
    # MongoDB cannot store numpy arrays directly
    vector_list = vector.tolist() if vector is not None else None
    
    # Update user document with new vector and weight
    user_db.update_one(
        {'_id': user_id}, 
        {'$set': {'vector': vector_list, 'total_weight': total_weight}}
    )


# ============================================================================
# PART 7: INCREMENTAL USER VECTOR UPDATES
# ============================================================================

def update_user_vector(old_vector, old_weight, item_vector, item_weight):
    """
    Incrementally update user vector when a new interaction is added.
    
    Instead of recalculating from scratch, this efficiently updates the
    existing vector with a new interaction. Useful for real-time updates.
    
    Algorithm:
    1. Compute weighted sum: old_vector * old_weight + item_vector * item_weight
    2. Update total weight: old_weight + item_weight
    3. Normalize the result
    
    Args:
        old_vector: Current user vector (numpy array)
        old_weight: Current total weight
        item_vector: Vector of the new item/event (numpy array)
        item_weight: Weight of the new interaction
    
    Returns:
        tuple: (new_normalized_vector, new_total_weight)
    """
    # Compute weighted sum
    if old_weight == 0:
        # First interaction: just use the item vector
        new_vector_sum = item_vector * item_weight
    else:
        # Combine old weighted vector with new weighted vector
        new_vector_sum = old_vector * old_weight + item_vector * item_weight

    # Update total weight
    new_weight = old_weight + item_weight

    # Normalize the result
    norm = np.linalg.norm(new_vector_sum)
    if norm == 0:
        new_vector = new_vector_sum
    else:
        new_vector = new_vector_sum / norm

    return new_vector, new_weight


def add_like(user_vector, weight, like):
    """
    Add a "like" interaction to the user vector incrementally.
    
    Args:
        user_vector: Current user vector (numpy array)
        weight: Current total weight
        like: Like interaction dict with 'vector' and 'ts' fields
    
    Returns:
        tuple: (updated_vector, updated_weight)
    """
    item_vec = np.array(like["vector"], dtype=float)
    ts = like["ts"]

    # Calculate interaction weight with time decay
    item_weight = compute_interaction_weight(ts, LIKE_STRENGTH)

    return update_user_vector(user_vector, weight, item_vec, item_weight)


def add_save(user_vector, weight, save):
    """
    Add a "save" interaction to the user vector incrementally.
    
    Args:
        user_vector: Current user vector (numpy array)
        weight: Current total weight
        save: Save interaction dict with 'vector' and 'ts' fields
    
    Returns:
        tuple: (updated_vector, updated_weight)
    """
    item_vec = np.array(save["vector"], dtype=float)
    ts = save["ts"]

    # Calculate interaction weight with time decay
    item_weight = compute_interaction_weight(ts, SAVE_STRENGTH)

    return update_user_vector(user_vector, weight, item_vec, item_weight)


def add_visit(user_vector, weight, visit):
    """
    Add a "visit" interaction to the user vector incrementally.
    
    Args:
        user_vector: Current user vector (numpy array)
        weight: Current total weight
        visit: Visit interaction dict with 'vector' and 'ts' fields
    
    Returns:
        tuple: (updated_vector, updated_weight)
    """
    item_vec = np.array(visit["vector"], dtype=float)
    ts = visit["ts"]

    # Calculate interaction weight with time decay
    item_weight = compute_interaction_weight(ts, VISIT_STRENGTH)

    return update_user_vector(user_vector, weight, item_vec, item_weight)


# ============================================================================
# PART 8: SIMILARITY SEARCH FUNCTIONS
# ============================================================================

def get_top_similar_events(vector, n=10, num_candidates=200, user_id=None, max_fetch_multiplier=5):
    """
    Find top N events most similar to the given vector using vector search.
    
    Uses MongoDB's $vectorSearch aggregation for efficient similarity search.
    This is content-based recommendation: finds events similar to a query vector.
    
    Args:
        vector: Query vector (list or numpy array)
        n: Number of results to return
        num_candidates: Number of candidates to consider (higher = more accurate, slower)
        user_id: If provided, excludes events this user has seen (from get_user_seen_event_ids)
        max_fetch_multiplier: Maximum multiplier for fetch_limit (e.g., 5 = fetch up to n*5 events)
    
    Returns:
        List of tuples: [(event_id, similarity_score), ...] sorted by similarity
        May return fewer than n results if user has seen most events
    """
    # Get excluded events if user_id is provided
    exclude_event_ids = None
    if user_id is not None:
        seen_events = get_user_seen_event_ids(user_id)
        # Convert to set for O(1) lookup
        exclude_event_ids = set(seen_events) if seen_events else set()
    
    # If we need to exclude events, fetch more candidates to account for filtering
    # Start with 2x, but can increase if needed
    fetch_limit = n * 2 if exclude_event_ids else n
    max_fetch = n * max_fetch_multiplier
    
    results = []
    current_fetch = 0
    
    # Keep fetching until we have enough results or hit max
    while len(results) < n and current_fetch < max_fetch:
        # Calculate how many more we need
        remaining = n - len(results)
        # Fetch more than needed to account for potential exclusions
        batch_size = min(remaining * 2, max_fetch - current_fetch)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": vector,
                    "path": "vector",
                    "numCandidates": num_candidates,
                    "limit": batch_size,
                    "index": "vector_index_events"
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "vector": 1,
                }
            }
        ]
        
        # Track which events we've already processed to avoid duplicates
        processed_ids = {event_id for event_id, _ in results}
        
        for doc in data_db.aggregate(pipeline):
            event_id = doc["_id"]
            
            # Skip if already in results (duplicate from vector search)
            if event_id in processed_ids:
                continue
            
            processed_ids.add(event_id)
            
            # Skip excluded events
            if exclude_event_ids and event_id in exclude_event_ids:
                continue
            
            # Calculate exact cosine similarity for ranking
            similarity = cosine_similarity(vector, doc["vector"])
            results.append((event_id, similarity))
            
            # Stop if we have enough results
            if len(results) >= n:
                break
        
        current_fetch += batch_size
        
        # If we didn't get any new results, break to avoid infinite loop
        if batch_size == 0:
            break
    
    return results


def get_top_similar_users(vector, n=10, num_candidates=200):
    """
    Find top N users most similar to the given vector using vector search.
    
    Uses MongoDB's $vectorSearch aggregation for efficient similarity search.
    Used for collaborative filtering: finds users with similar preferences.
    
    Args:
        vector: Query vector (list or numpy array) - typically a user's preference vector
        n: Number of results to return
        num_candidates: Number of candidates to consider (higher = more accurate, slower)
    
    Returns:
        List of tuples: [(user_id, similarity_score), ...] sorted by similarity
    """
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": vector,
                "path": "vector",  # Field containing user vectors
                "numCandidates": num_candidates,
                "limit": n,
                "index": "vector_index_user"  # Vector search index name
            }
        },
        {
            "$project": {
                "_id": 1,
                "vector": 1,
            }
        }
    ]

    results = []
    for doc in user_db.aggregate(pipeline):
        # Use dot product for normalized vectors (equivalent to cosine similarity)
        similarity = dot_similarity(vector, doc["vector"])
        results.append((doc["_id"], similarity))

    return results


# ============================================================================
# PART 9: COLLABORATIVE FILTERING RECOMMENDATIONS
# ============================================================================

def collaborative_filtering_recommendations(
    user_id,
    n=10,
    num_similar_users=20,
    min_similarity=0.0,
    exclude_interacted=True
):
    """
    Generate collaborative filtering recommendations for a user.
    
    Algorithm:
    1. Find similar users based on user vector similarity
    2. Aggregate events from similar users, weighted by:
       - User similarity score (how similar the user is)
       - Interaction type weight (like/save/visit strength)
    3. Filter out events the user has already interacted with
    4. Return top N recommendations by aggregated score
    
    This is user-based collaborative filtering: "users like you also liked..."
    
    Args:
        user_id: ObjectId of the target user
        n: Number of recommendations to return
        num_similar_users: Number of similar users to consider
        min_similarity: Minimum similarity threshold (0-1) to filter similar users
        exclude_interacted: If True, exclude events user has already interacted with
    
    Returns:
        List of tuples: [(event_id, score), ...] sorted by score descending
        Returns empty list if user has no vector or no similar users found
    """
    # Get target user's vector
    user = user_db.find_one({'_id': user_id}, {'vector': 1})
    if user is None or user.get('vector') is None:
        return []
    
    user_vector = user['vector']
    
    # Get events user has already interacted with (to exclude from recommendations)
    excluded_event_ids = set()
    if exclude_interacted:
        excluded_event_ids = get_user_seen_event_ids(user_id)
    
    # Find similar users using vector search
    similar_users = get_top_similar_users(
        user_vector,
        n=num_similar_users,
        num_candidates=200
    )
    
    # Filter by minimum similarity threshold
    similar_users = [(uid, sim) for uid, sim in similar_users if sim >= min_similarity]
    
    if not similar_users:
        return []
    
    # Aggregate events from similar users with weighted scores
    event_scores = {}  # event_id -> total_score
    
    for similar_user_id, similarity_score in similar_users:
        # Get events this similar user interacted with
        user_events = get_events_from_user(similar_user_id)
        
        for event_id, interaction_type, interaction_weight in user_events:
            # Skip if target user already interacted with this event
            if event_id in excluded_event_ids:
                continue
            
            # Calculate weighted score:
            # similarity_score * interaction_weight
            # - similarity_score: how similar the user is (0-1)
            # - interaction_weight: strength of interaction type (like/save/visit)
            score = similarity_score * interaction_weight
            
            # Aggregate scores (sum across all similar users)
            # Events liked by more similar users get higher scores
            if event_id not in event_scores:
                event_scores[event_id] = 0.0
            event_scores[event_id] += score
    
    # Sort by score and return top N
    sorted_events = sorted(
        event_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_events[:n]


# ============================================================================
# PART 10: HYBRID RECOMMENDATION SYSTEM
# ============================================================================

def hybrid_recommendations(
    user_id,
    n=10,
    cf_weight=0.5,
    content_weight=0.5,
    num_similar_users=20
):
    """
    Hybrid recommendation combining collaborative filtering and content-based.
    
    Combines two recommendation approaches:
    1. Collaborative Filtering: Based on similar users' preferences
    2. Content-Based: Based on similarity to user's preference vector
    
    The scores from both approaches are normalized and weighted, then combined.
    
    Args:
        user_id: ObjectId of the target user
        n: Number of recommendations to return
        cf_weight: Weight for collaborative filtering scores (0-1)
                   Higher = more emphasis on what similar users like
        content_weight: Weight for content-based scores (0-1)
                        Higher = more emphasis on content similarity
        num_similar_users: Number of similar users for CF component
    
    Returns:
        List of tuples: [(event_id, final_score), ...] sorted by score descending
        Returns empty list if user has no vector
    """
    # Get collaborative filtering recommendations
    # Get more candidates (n*2) to have better pool for hybrid combination
    cf_recs = dict(collaborative_filtering_recommendations(
        user_id,
        n=n * 2,
        num_similar_users=num_similar_users
    ))
    
    # Get content-based recommendations (using user vector)
    user = user_db.find_one({'_id': user_id}, {'vector': 1})
    if user is None or user.get('vector') is None:
        return []
    
    content_recs = dict(get_top_similar_events(
        user['vector'],
        n=n * 2,
        user_id=user_id # Pass user_id to exclude seen events
    ))
    
    # Get all unique event IDs from both recommendation sets
    all_event_ids = set(cf_recs.keys()) | set(content_recs.keys())
    
    if not all_event_ids:
        return []
    
    # Normalize CF scores to [0, 1] range for fair combination
    if cf_recs:
        max_cf = max(cf_recs.values())
        min_cf = min(cf_recs.values())
        cf_range = max_cf - min_cf if max_cf != min_cf else 1.0
        cf_recs = {eid: (score - min_cf) / cf_range for eid, score in cf_recs.items()}
    
    # Normalize content scores to [0, 1] range
    # Cosine similarity is [-1, 1], so we normalize it
    if content_recs:
        max_content = max(content_recs.values())
        min_content = min(content_recs.values())
        content_range = max_content - min_content if max_content != min_content else 1.0
        content_recs = {eid: (score - min_content) / content_range 
                       for eid, score in content_recs.items()}
    
    # Combine normalized scores with weights
    hybrid_scores = {}
    for event_id in all_event_ids:
        cf_score = cf_recs.get(event_id, 0.0) * cf_weight
        content_score = content_recs.get(event_id, 0.0) * content_weight
        hybrid_scores[event_id] = cf_score + content_score
    
    # Sort by combined score and return top N
    sorted_events = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_events[:n]


# ============================================================================
# PART 11: MAIN / TEST CODE
# ============================================================================

if __name__ == '__main__':
    """
    Example usage and testing code.
    
    This section demonstrates how to use the recommendation functions.
    """
    # Example: Recalculate a user's vector
    test_user_id = ObjectId("692223968f85b548309145d3")
    recalc_user_vector(test_user_id)
    
    # Print updated user document
    # print(user_db.find_one({'_id': test_user_id}))
    
    # Example: Get collaborative filtering recommendations
    # cf_recs = collaborative_filtering_recommendations(
    #     user_id=test_user_id,
    #     n=10,
    #     num_similar_users=20,
    #     min_similarity=0.1
    # )
    # print("CF Recommendations:", cf_recs)
    
    # Example: Get hybrid recommendations
    hybrid_recs = hybrid_recommendations(
        user_id=test_user_id,
        n=10,
        cf_weight=0.6,      # 60% collaborative filtering
        content_weight=0.4   # 40% content-based
    )
    print("Hybrid Recommendations:", hybrid_recs)
