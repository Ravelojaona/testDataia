# RAG Madagascar — Test Technique AI Engineer

Système RAG de bout en bout sur la page Wikipedia **Madagascar**, avec architecture hexagonale propre, API REST, CLI et suite de tests.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ADAPTERS (entrées/sorties)                              │
│  ┌──────────────────┐   ┌──────────────────────────┐    │
│  │   FastAPI /query  │   │   CLI (click) rag query  │    │
│  └────────┬─────────┘   └────────────┬─────────────┘    │
└───────────┼─────────────────────────┼────────────────────┘
            │ dépend de               │
┌───────────▼─────────────────────────▼────────────────────┐
│  APPLICATION (use cases)                                  │
│  ┌─────────────────────┐  ┌──────────────────────────┐   │
│  │  BuildIndexUseCase  │  │     QueryUseCase          │   │
│  └─────────────────────┘  └──────────────────────────┘   │
│  dépend uniquement des ports (interfaces)                 │
└────────────────────────────────────────────────────────────┘
            │ implémentés par
┌───────────▼────────────────────────────────────────────────┐
│  INFRASTRUCTURE (dépendances externes)                      │
│  WikipediaLoader │ WikipediaChunker │ OpenAIEmbedder        │
│  HybridRetriever (FAISS + BM25 + RRF) │ OpenAIGenerator     │
└────────────────────────────────────────────────────────────┘
            │ modèles définis dans
┌───────────▼────────────────────────────────────────────────┐
│  DOMAIN (cœur métier, zéro dépendance externe)             │
│  Chunk │ QueryResult │ SearchResult │ IndexStats            │
│  IDocumentLoader │ IChunker │ IEmbedder │ IRetriever        │
│  IGenerator │ ITranslator                                   │
└────────────────────────────────────────────────────────────┘
```

---

## Choix techniques

| Composant | Choix | Justification |
|-----------|-------|---------------|
| Ingestion | Wikipedia Action API | Plus stable que le scraping HTML brut |
| Chunking texte | Sliding window 500 tok / 50 overlap | Contexte suffisant, overlap évite les coupures |
| Chunking tableaux | Chunk atomique | Un tableau perd son sens si coupé |
| Embeddings | `text-embedding-3-small` | Meilleur ratio coût/qualité, 1 536d |
| Vector store | FAISS IndexFlatIP | Recherche exacte, corpus < 1 000 chunks |
| Sparse | BM25 Okapi | Capture les termes exacts (dates, noms) |
| Fusion | RRF (k=60) | Agnostique aux échelles de scores |
| LLM | GPT-4o, temp=0 | Cohérence factuelle maximale |
| Cross-lingual | Détection FR + traduction GPT-4o-mini | Indices en anglais, questions en FR/EN |

---

## Démarrage rapide

### Avec Docker (recommandé)

```bash
cp .env.example .env
# Édite .env : OPENAI_API_KEY=sk-...

# 1. Construire l'image
docker compose build

# 2. Construire l'index (~0.02$, une seule fois)
docker compose run --rm indexer

# 3. Démarrer l'API
docker compose up api

# L'API est disponible sur http://localhost:8000
# Swagger UI : http://localhost:8000/docs
```

### Sans Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env  # puis ajoute ta clé

# Construire l'index
python -m src.adapters.cli.cli build

# Démarrer l'API
uvicorn src.adapters.api.app:app --reload

# CLI
python -m src.adapters.cli.cli query "What is the capital of Madagascar?"
python -m src.adapters.cli.cli query "Quelle est la superficie ?"
python -m src.adapters.cli.cli interactive
```

---

## API REST

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Statut + index prêt ? |
| `GET` | `/stats` | Métadonnées de l'index |
| `POST` | `/index/build` | Construire/recharger l'index |
| `POST` | `/query` | Poser une question |

**Exemple `/query` :**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of Madagascar?", "top_k": 5}'
```

---

## Tests

```bash
# Tests unitaires (pas de clé API requise)
pytest tests/unit/ -v

# Tests unitaires + couverture
pytest tests/unit/ --cov=src --cov-report=term-missing

# Tests d'intégration (requiert OPENAI_API_KEY + index pré-construit)
pytest tests/integration/ -v -m integration

# Via Docker
docker compose run --rm tests
```

---

## Structure du projet

```
rag_madagascar/
├── src/
│   ├── domain/              # Entités + interfaces (zéro dépendance)
│   │   ├── models.py
│   │   └── ports.py
│   ├── infrastructure/      # Implémentations concrètes
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── hybrid_retriever.py
│   │   └── generator.py
│   ├── application/         # Use cases (orchestration)
│   │   ├── index_usecase.py
│   │   └── query_usecase.py
│   └── adapters/            # Interfaces utilisateur
│       ├── api/             # FastAPI
│       │   ├── app.py
│       │   ├── routes.py
│       │   └── schemas.py
│       └── cli/             # Click CLI
│           └── cli.py
├── tests/
│   ├── unit/                # Tests sans API (mocks)
│   │   ├── test_models.py
│   │   ├── test_chunker.py
│   │   ├── test_retriever.py
│   │   └── test_query_usecase.py
│   ├── integration/         # Tests end-to-end (requiert clé API)
│   │   └── test_pipeline.py
│   └── conftest.py
├── evaluate.py              # 17 questions d'évaluation
├── Dockerfile               # Multi-stage build
├── docker-compose.yml       # api + indexer + tests services
├── pyproject.toml           # Config pytest + entry points
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

---

## Réflexion — 2 semaines vs 3 jours

Avec deux semaines plutôt que trois jours, l'effort irait d'abord vers la fiabilité de la mesure plutôt que vers de nouvelles fonctionnalités. L'évaluation actuelle repose sur une correspondance de mots-clés et sur la détection d'une phrase de refus fixe, ce qui est fragile dès que le modèle reformule légèrement sa réponse ou répond en français ; je remplacerais cette approche par un LLM-as-judge comparant chaque réponse à une réponse de référence, et j'ajouterais un re-ranker cross-encoder (`ms-marco-MiniLM`) pour vérifier si le gain de précision sur le top-k justifie la latence supplémentaire. Je traiterais aussi deux limites connues du pipeline : le parsing des tableaux avec `rowspan`/`colspan`, qui dégrade actuellement le rendu Markdown de certains tableaux administratifs, et l'agrégation explicite des valeurs historiques (population, présidents) plutôt que la simple citation de passages disjoints.

Sur le plan produit, je pousserais le système vers un usage réel plutôt qu'un simple harness de test : une interface web légère (Streamlit) pour permettre des tests non-techniques, un monitoring des coûts et latences API en production, et un pipeline CI/CD exécutant les tests unitaires et un sous-ensemble de l'évaluation à chaque commit pour détecter les régressions de qualité de réponse, pas seulement les régressions de code. J'envisagerais enfin d'étendre le périmètre à plusieurs pages Wikipedia liées (par exemple les pages des régions administratives), ce qui obligerait à revoir la stratégie de chunking et de citation pour gérer plusieurs sources distinctes sans ambiguïté.

---

## Limites connues

1. **Fraîcheur** : contenu téléchargé une seule fois, pas de refresh automatique.
2. **Tableaux complexes** : `rowspan`/`colspan` peut dégrader le Markdown.
3. **Désambiguïsation temporelle** : le système cite les passages mais n'agrège pas les valeurs historiques.
4. **Détection de langue** : heuristique sur mots fonctionnels, peut échouer sur phrases très courtes.
