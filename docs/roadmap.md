# Roadmap del proyecto

## Entregable 1 — MVP baseline

Objetivo: reproducir un pipeline básico tipo Document RAG multimodal.

```text
PDFs → páginas como imágenes → retrieval visual → VLM → respuesta
```

Componentes:
- `pdf2image`
- CLIP + FAISS
- Qwen2-VL

## Entregable 2 — Parsing estructurado

Objetivo: extraer estructura documental.

Opciones:
- Docling
- Marker
- Unstructured

Salida esperada:

```json
{
  "doc_id": "report_01",
  "page": 1,
  "blocks": [
    {"type": "title", "text": "...", "bbox": [0, 0, 100, 30]},
    {"type": "table", "text": "...", "bbox": [0, 40, 500, 300]}
  ]
}
```

## Entregable 3 — Índice jerárquico

Representar:

```text
documento → página → bloque → región
```

Metadata sugerida:

```text
document_id
page_id
block_id
region_id
modality
bbox
text
image_path
embedding
```

## Entregable 4 — Política adaptativa inicial

Reglas iniciales:

```text
Pregunta sobre tabla → priorizar regiones de tabla
Pregunta sobre figura/gráfico → priorizar visual retrieval
Pregunta sobre título/encabezado → priorizar layout-aware retrieval
Pregunta sobre fecha/nombre/total → priorizar text/key-value retrieval
```

## Entregable 5 — Evaluación

Comparar:

1. Chunking fijo + text retrieval.
2. Page-level visual retrieval.
3. Hierarchical retrieval.
4. Adaptive segmentation + adaptive retrieval.

Métricas:
- Recall@k
- MRR
- Accuracy
- ANLS
- tiempo de respuesta
