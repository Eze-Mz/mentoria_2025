# Notebooks

Las notebooks numeradas son la version curada para portfolio y lectura publica. Mantienen el orden del trabajo original, pero con rutas reproducibles, menos ruido de Colab y artefactos derivados para comparar modelos.

| Notebook | Rol |
| --- | --- |
| `01_exploracion_inicial.ipynb` | exploracion del corpus, fechas, provincias y volumen |
| `02_curacion_features.ipynb` | limpieza, transformaciones y familias de variables |
| `03a_clustering_kmeans.ipynb` | KMeans con TF-IDF, Word2Vec y Sentence-BERT; terminos representativos por cluster |
| `03b_topic_modeling_clasico.ipynb` | NMF y LDA con barridos de topicos, metricas y ejemplos representativos |
| `03c_bertopic_iteraciones.ipynb` | tres variantes de BERTopic con embeddings cacheados, metricas y visualizaciones |
| `05_sentimiento_hostilidad.ipynb` | sentimiento, hostilidad y cruces por topico |

`archive/` conserva entregas originales y exploraciones personales. No es la entrada recomendada para recorrer el proyecto, pero queda como trazabilidad del proceso.

## Entorno

Crear y activar el entorno desde la raiz del repo:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
jupyter lab
```

Usar el kernel/interprete de `.venv`. Si no aparece en Jupyter, registrar:

```bash
python -m ipykernel install --user --name mentoria-obesidad --display-name "Python (mentoria-obesidad)"
```

## Datos

Las notebooks esperan los CSV completos en `data/raw/` y `data/processed/`. Esos archivos son locales e ignorados por Git. La demo publica usa un payload sanitizado en `site/data/`, que no contiene textos completos ni identificadores.
