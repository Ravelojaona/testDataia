# Audit de conformité — Test technique AI Engineer (RAG Madagascar)

Document de vérification du projet par rapport au cahier des charges
(`testDATA.pdf`). Réalisé le 2026-07-28, en relisant le code réel puis en
exécutant les tests et une partie du pipeline.

## Résumé

| # | Exigence | Statut |
|---|----------|--------|
| 1 | Ingestion | ✅ Conforme |
| 2 | Nettoyage | ✅ Conforme |
| 3 | Chunking (texte + tableaux) | ✅ Conforme |
| 4 | Indexation & Retrieval | ✅ Conforme (re-ranking optionnel non fait) |
| 5 | Génération (grounding, citation, refus) | ✅ Conforme |
| 6 | Cross-lingual FR/EN | ✅ Corrigé pendant l'audit |
| 7 | Évaluation (≥15 questions + résultats) | ⚠️ Harness conforme, résultats non exécutés |
| 8 | Livrables (README, rapport de réflexion) | ✅ Corrigé pendant l'audit |

## Détail

### 1. Ingestion — conforme
`src/infrastructure/loader.py` interroge l'API MediaWiki (`action=parse`)
plutôt que du scraping HTML brut — choix documenté (plus stable face aux
changements de mise en page). Un bug bloquant a été trouvé et corrigé
pendant l'audit : Wikipedia renvoyait `403 Forbidden` car aucun
`User-Agent` n'était envoyé (politique Wikimedia). Un en-tête `User-Agent`
descriptif a été ajouté (`loader.py`). Après correction, l'ingestion
fonctionne : 203 chunks générés dont 4 chunks de type tableau.

### 2. Nettoyage — conforme
`chunker.py::_clean` supprime les marqueurs `[1]`, `[note 1]`,
`[citation needed]`, et le parsing HTML retire `<sup>`, `<style>`,
`<script>` ainsi que les éléments `reference`/`mw-editsection`/`navbox`/
`thumb` avant extraction du texte.

### 3. Chunking — conforme
Fenêtre glissante de 500 tokens avec chevauchement de 50 (tokenisation
`tiktoken`), découpage aligné sur les sections H2/H3/H4 avec préfixe
`[Section: ...]`. Les tableaux (régions, groupes ethniques) sont traités
comme des chunks atomiques convertis en Markdown, pour rester
interrogeables sans être coupés en morceaux incohérents. Limite assumée :
pas de gestion `rowspan`/`colspan` (documentée dans les limites connues).

### 4. Indexation & Retrieval — conforme
Embeddings `text-embedding-3-small` (justifié coût/qualité). Vector store
FAISS `IndexFlatIP` (justifié par la taille réduite du corpus, <1000
chunks). Recherche hybride dense + BM25 fusionnée par Reciprocal Rank
Fusion (`hybrid_retriever.py`). Le re-ranking cross-encoder mentionné
comme option dans le cahier des charges n'est pas implémenté ; c'est
listé comme piste d'amélioration dans le README, ce qui est acceptable
puisque le sujet le présente comme optionnel.

### 5. Génération — conforme
Prompt système forçant l'usage exclusif du contexte fourni, citation
obligatoire (`Source: [Section] | Chunk #[id]`) en fin de réponse, et
phrase de refus fixe et explicite quand l'information est absente. La
désambiguïsation temporelle (plusieurs présidents, plusieurs recensements)
est traitée par une règle de prompt dédiée.

### 6. Cross-lingual — corrigé pendant l'audit
Détection FR/EN heuristique + traduction de la requête vers l'anglais
avant recherche (l'index est en anglais). **Manque identifié** : rien
n'imposait explicitement au modèle de répondre dans la langue de la
question. Correction apportée : ajout d'une règle 6 dans le prompt système
(`generator.py`) — *"Always answer in the SAME language as the
question"* — sans casser la phrase de refus fixe en anglais utilisée par
`evaluate.py` pour détecter les refus (les deux règles sont compatibles,
la phrase de refus reste un cas particulier volontairement figé).

### 7. Évaluation — harness conforme, résultats non produits
`evaluate.py` contient bien 17 questions couvrant les 7 catégories
demandées (fait simple, chiffre précis, lecture de tableau, raisonnement
multi-passages, ambiguïté temporelle, hors périmètre, partiellement
couverte), avec calcul du taux de bonnes réponses, du taux de faux
positifs sur les questions pièges et du temps de réponse, exporté en JSON.

**Point bloquant constaté pendant l'audit** : la clé `OPENAI_API_KEY`
présente dans `.env` n'a plus de quota disponible (`insufficient_quota`
sur l'appel d'embeddings). L'ingestion et le chunking se sont exécutés
avec succès (203 chunks), mais la génération d'embeddings et donc
l'évaluation complète n'ont pas pu tourner de bout en bout. **Aucun
`eval_results.json` n'a donc pu être produit avec cette clé.** À refaire
avec une clé disposant de crédit avant la remise finale — c'est le seul
livrable réellement manquant.

### 8. Livrables — corrigé pendant l'audit
Code source complet et exécutable (architecture hexagonale, CLI, API
REST, Dockerfile/docker-compose), README détaillé (architecture, choix
techniques, limites connues). **Manque identifié** : le rapport de
réflexion "2 semaines vs 3 jours" était rédigé en listes à puces alors que
le cahier des charges demande explicitement 2 paragraphes de prose.
Corrigé : le README contient maintenant 2 paragraphes de prose continue à
la place des listes.

## Ce qui reste à faire avant la remise

1. Relancer `python evaluate.py` avec une clé OpenAI disposant de crédit,
   puis committer `eval_results.json` (ou coller le tableau de résultats
   dans le README).
2. Vérifier que le rendu final des réponses en français fonctionne bien
   avec la nouvelle règle de prompt (non testable sans quota API pendant
   cet audit).
3. Rendre le dépôt GitHub public (déjà poussé sur
   `github-mahery:Ravelojaona/testDataia.git`, branche `main`) — vérifier
   la visibilité du repo avant d'envoyer le lien.
