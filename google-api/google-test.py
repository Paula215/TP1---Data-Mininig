import requests
import json
import os
import time


API_KEY = "YOUR_GOOGLE_API_KEY"  # Reemplaza con tu clave de API de Google
CITY = "Lima, Peru"

# ============================================
# CONFIGURACIÓN
# ============================================
DISTRICTS = [
    "Barranco", "Miraflores", "Chorrillos", "San Isidro", "San Miguel",
    "La Molina", "Santiago de Surco", "San Borja", "Lince", "Jesús María",
    "Pueblo Libre", "Magdalena", "Callao", "San Martín de Porres",
    "Los Olivos", "Comas", "Ate", "Rímac", "Cercado de Lima", "Breña", "Lurín", 
    "Pachacámac", "Villa El Salvador", "Villa María del Triunfo","Chaclacayo",
    "Cieneguilla", "San Juan de Lurigancho", "San Juan de Miraflores", "Independencia",
    "Carabayllo", "Puente Piedra", "Santa Anita", "El Agustino", "La Victoria", "Surquillo",
    "Barranca",
]

KEYWORDS = [
    "playa", "iglesia histórica", "zoologico", "malecon", "fortaleza","galeria de arte",
    "Museo", "sitio turístico", "sitio arqueológico", "Centro Cultural", "laguna", "plaza historica",
    "mirador", "parque de diversiones", "acuario", "parque temático", "catedral","patrimonio cultural",
]

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def geocode_district(district_name):
    """Obtiene coordenadas (lat, lng) del distrito de Lima Metropolitana."""
    # Agrega contexto explícito: provincia y país
    address = f"{district_name}, Provincia de Lima, Departamento de Lima, Perú"
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&region=pe&key={API_KEY}"

    try:
        res = requests.get(url)
        data = res.json()

        if data.get("status") != "OK" or not data["results"]:
            print(f"⚠️ Geocoding falló para {district_name} ({data.get('status')})")
            return None, None

        loc = data["results"][0]["geometry"]["location"]
        print(f"  📍 Coordenadas {district_name}: ({loc['lat']:.5f}, {loc['lng']:.5f})")
        return loc["lat"], loc["lng"]

    except Exception as e:
        print(f"⚠️ Error al geocodificar {district_name}: {e}")
        return None, None



def fetch_places(query, lat, lng, radius=3000):
    """Llama a la API de Places Nearby Search."""
    url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        f"keyword={query}&location={lat},{lng}&radius={radius}&key={API_KEY}"
    )
    res = requests.get(url)
    data = res.json()
    results = data.get("results", [])
    next_page_token = data.get("next_page_token")

    while next_page_token:
        time.sleep(2)
        res = requests.get(
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
            f"pagetoken={next_page_token}&key={API_KEY}"
        )
        data = res.json()
        results.extend(data.get("results", []))
        next_page_token = data.get("next_page_token")
    return results


def fetch_place_details(place_id):
    """Obtiene detalles de cada lugar (reviews, fotos, dirección, etc.)."""
    fields = "name,rating,formatted_address,geometry,photos,reviews,types,url,vicinity"
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields={fields}&key={API_KEY}"
    res = requests.get(url)
    return res.json().get("result", {})

def save_results(district, keyword, data):
    """Guarda resultados por keyword y distrito, omitiendo vacíos."""
    if not data:
        print(f"⚠️ No se encontraron lugares para {keyword} en {district}. No se guardará archivo.")
        return

    base_dir = os.path.join("results", keyword.lower().replace(" ", "_"))
    os.makedirs(base_dir, exist_ok=True)

    file_name = f"{district.lower().replace(' ', '_')}_{keyword.lower().replace(' ', '_')}.json"
    path = os.path.join(base_dir, file_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Guardado {len(data)} lugares en {path}")

# ============================================
# PROCESO PRINCIPAL
# ============================================

def run_extraction():
    for keyword in KEYWORDS:
        print("\n" + "=" * 60)
        print(f"🔍 Keyword: {keyword}")
        print("=" * 60)

        for district in DISTRICTS:
            print(f"\n🌎 Extrayendo lugares en distrito: {district} ...")
            lat, lng = geocode_district(district)
            if not lat:
                print(f"⚠️ No se pudo geocodificar {district}")
                continue

            raw_places = fetch_places(keyword, lat, lng)
            filtered_places = []
            seen_ids = set()

            for p in raw_places:
                place_id = p.get("place_id")
                if not place_id or place_id in seen_ids:
                    continue
                types = p.get("types", [])
                if any(t in ["parking", "car_rental", "gas_station"] for t in types):
                    continue  # ❌ excluye estos tipos


                details = fetch_place_details(place_id)
                address = (details.get("formatted_address") or "").lower()
                vicinity = (details.get("vicinity") or "").lower()

                # Filtro textual para asegurar que pertenece al distrito
                if district.lower() not in address and district.lower() not in vicinity:
                    continue

                seen_ids.add(place_id)
                filtered_places.append({
                    "name": details.get("name"),
                    "address": details.get("formatted_address"),
                    "rating": details.get("rating"),
                    "types": details.get("types"),
                    "location": details.get("geometry", {}).get("location"),
                    "url": details.get("url"),
                    "photos": [
                        f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=800&photo_reference={ph['photo_reference']}&key={API_KEY}"
                        for ph in details.get("photos", [])
                    ] if details.get("photos") else [],
                    "reviews": [
                        {"author": r.get("author_name"), "rating": r.get("rating"),
                         "text": r.get("text"), "time": r.get("relative_time_description")}
                        for r in details.get("reviews", [])
                    ] if details.get("reviews") else []
                })

                print(f"  → {details.get('name')} ✅ ({district})")

            save_results(district, keyword, filtered_places)
            time.sleep(2)  # evita superar límites de Google API

    print("\n🎯 Extracción completada correctamente para todas las keywords y distritos.")


# ============================================
# EJECUCIÓN
# ============================================

if __name__ == "__main__":
    run_extraction()
