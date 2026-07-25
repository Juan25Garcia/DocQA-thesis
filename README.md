# docqa-thesis-mvp

MVP inicial para tesis: **Document RAG multimodal sobre PDFs**.

El objetivo de este primer entregable es tener un pipeline mínimo:

```text
PDFs
  ↓
Páginas como imágenes
  ↓
Índice visual por página
  ↓
Pregunta del usuario
  ↓
Top-k páginas relevantes
  ↓
LLM/VLM
  ↓
Respuesta fundamentada
```

Este repo **no intenta implementar todavía la propuesta completa** de segmentación adaptativa. Primero construimos un baseline funcional tipo M3DocRAG simplificado. Después agregaremos parsing estructurado, índice jerárquico y política adaptativa.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

En Linux/WSL, `pdf2image` necesita Poppler:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

## Uso rápido

Coloca PDFs en:

```text
data/pdfs/
```

Convierte páginas:

```bash
python -m src.pdf_to_images
```

Crea metadata:

```bash
python -m src.create_metadata
```

Construye índice visual con CLIP + FAISS:

```bash
python -m src.build_clip_index
```

Prueba retrieval:

```bash
python -m src.retrieve_clip --question "What is the total amount?" --top-k 5
```

Demo end-to-end con Qwen2-VL:

```bash
python -m src.demo --question "What is the total amount?" --top-k 3
```

Si no quieres correr generación todavía:

```bash
python -m src.demo --question "What is the total amount?" --top-k 3 --retrieval-only
```

## Estado

- [x] PDF → imágenes
- [x] metadata de páginas
- [x] retrieval visual básico con CLIP + FAISS
- [x] demo con Qwen2-VL
- [ ] ColPali / ColQwen retrieval
- [ ] Docling / Marker parsing
- [ ] índice jerárquico
- [ ] política adaptativa de segmentación

## Conexión con tesis

Baseline:

```text
PDF → page images → page retrieval → VLM answer
```

Propuesta futura:

```text
PDF → OCR + layout + visual structure
    → adaptive segmentation
    → hierarchical multimodal index
    → adaptive retrieval
    → grounded answer
```
