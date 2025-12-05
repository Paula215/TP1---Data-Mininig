# Sistema de Recomendación de Lugares Turísticos con IA

Este proyecto implementa un sistema de recomendación inteligente para lugares turísticos utilizando procesamiento de lenguaje natural (LLM) y bases de datos vectoriales (ChromaDB).

El sistema consta de dos componentes principales:
1.  **Pipeline de Datos (`pipeline.py`)**: Procesa reseñas y descripciones de lugares desde archivos JSON, los estructura, genera embeddings y puebla la base de datos vectorial.
2.  **Aplicación Web (`app.py`)**: Una interfaz en Streamlit que permite a los usuarios recibir recomendaciones personalizadas basadas en sus preferencias de viaje.

## Requisitos Previos

Asegúrate de tener Python instalado (se recomienda versión 3.10 o superior).

1.  Clona este repositorio o ubícate en la carpeta del proyecto.
2.  Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Configuración

Antes de ejecutar el sistema, necesitas configurar tu clave de API de OpenRouter, ya que el sistema utiliza modelos de lenguaje para procesar las reseñas y generar recomendaciones.

1.  Abre el archivo `pipeline.py`.
2.  Busca la variable `OPENROUTER_API_KEY` (cerca del inicio del archivo) y coloca tu clave:
    ```python
    OPENROUTER_API_KEY = "sk-or-v1-..."
    ```
3.  Abre el archivo `app.py`.
4.  Verifica o actualiza la variable `OPENROUTER_API_KEY` con tu clave.

## Guía de Uso

### Paso 1: Poblar la Base de Datos Vectorial

Para que el sistema pueda recomendar lugares, primero debe procesar la información existente en la carpeta `tripadvisor_extractions` y generar los "embeddings" (representaciones matemáticas del texto) en ChromaDB.

Ejecuta el siguiente comando en tu terminal para correr el pipeline completo (procesamiento de reseñas + generación de base de datos):

```bash
python -c "from pipeline import run_full_pipeline; run_full_pipeline(process_reviews=True, generate_db=True)"
```

**¿Qué hace este comando?**
*   **Lee** los archivos JSON de la carpeta `tripadvisor_extractions`.
*   **Estructura** la información desordenada usando un LLM (Generando `reviews_structured.csv`).
*   **Genera Embeddings** usando el modelo `nomic-ai/nomic-embed-text-v2-moe`.
*   **Guarda** los vectores en la carpeta local `chroma_db`.

*Nota: La primera vez que ejecutes esto, se descargará el modelo de embeddings, lo cual puede tomar unos minutos dependiendo de tu conexión.*

### Paso 2: Desplegar la Aplicación

Una vez que la base de datos (`chroma_db`) ha sido poblada, puedes iniciar la interfaz de usuario.

Ejecuta:

```bash
streamlit run app.py
```

Esto abrirá una pestaña en tu navegador (usualmente en `http://localhost:8501`) donde podrás:
1.  Ingresar tus preferencias de viaje (intereses, compañía, presupuesto).
2.  Recibir recomendaciones de lugares turísticos basadas en la similitud semántica con tu búsqueda.

## Estructura del Proyecto

*   `app.py`: Aplicación principal de Streamlit.
*   `pipeline.py`: Lógica de extracción, transformación y carga (ETL) de datos.
*   `tripadvisor_extractions/`: Carpeta con los datos crudos (JSON) de los lugares.
*   `chroma_db/`: Carpeta donde se almacena la base de datos vectorial (generada automáticamente).
*   `requirements.txt`: Lista de librerías de Python necesarias.
