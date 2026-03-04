# ADR-0093: BERTopic Stopword Filtering for c-TF-IDF Topic Labels

**Date:** 2026-03-04
**Status:** Accepted

## Context

Phase 20 (Bill Text Analysis) uses BERTopic for topic modeling. BERTopic's c-TF-IDF layer extracts topic representations from a `CountVectorizer` internally. The default `CountVectorizer` has `stop_words=None`, meaning English function words and legislative boilerplate are included in the vocabulary.

This produced uninterpretable topic labels. First pass with English-only stopwords still showed labels like "Topic 0: shall, statuteref, person, section, amendments" — legislative boilerplate terms that appear in virtually every bill regardless of policy area. These are as meaningless as "the, of, and" for distinguishing topics.

The clustering itself (UMAP + HDBSCAN on semantic embeddings) is unaffected — only the c-TF-IDF label extraction needs filtering.

## Decision

Pass an explicit `CountVectorizer` to BERTopic with four settings:

### 1. Custom stopwords: English + legislative boilerplate

`LEGISLATIVE_STOPWORDS` extends scikit-learn's 318 English stopwords with 18 legislative boilerplate terms:

| Category | Terms |
|----------|-------|
| Mandatory legal language | `shall` |
| Preprocessing artifact | `statuteref` (normalized K.S.A. references) |
| Structural markers | `section`, `subsection`, `paragraph` |
| Amendatory boilerplate | `amendments`, `amendment`, `amended`, `amend` |
| Archaic legal connectors | `thereto`, `thereof`, `therein`, `herein`, `hereby`, `hereof` |
| Legal boilerplate | `pursuant`, `provision`, `provisions` |

Terms deliberately excluded from the list:
- **`state`, `kansas`** — appear in every bill but form useful bigrams ("state board", "kansas department"). Handled by `max_df` instead, which filters unigrams without blocking bigrams.
- **`person`, `act`, `means`, `law`** — can be topically meaningful in criminal law, consumer protection, or procedural topics.

### 2. Bigrams: `ngram_range=(1, 2)`

Legislative terminology is often multi-word ("tax credit", "school district", "motor vehicle"). Note: stopwords block bigrams containing those words (e.g., adding "shall" blocks "shall not"), which is desirable — those bigrams are also boilerplate.

### 3. Document frequency filter: `max_df=0.85`

Terms appearing in >85% of individual bills are filtered automatically. This catches domain-ubiquitous terms like "state" and "kansas" without needing them in the stopword list — preserving useful bigrams like "state board" and "kansas department".

### 4. Minimum frequency: `min_df=2`

Terms must appear in at least 2 bills. Filters hapax legomena (single-occurrence terms) that add noise to the vocabulary.

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

vectorizer_model = CountVectorizer(
    stop_words=ENGLISH_STOP_WORDS | LEGISLATIVE_STOPWORDS,
    ngram_range=(1, 2),
    min_df=2,
    max_df=VECTORIZER_MAX_DF,  # 0.85
)
```

## Consequences

### Benefits

- **Topic labels show policy content.** Instead of "shall, statuteref, person, section, amendments", topics show terms like "tax, property, income" or "school, education, board, school district".
- **Two-layer filtering.** Curated stopwords for known boilerplate + `max_df` for data-driven ubiquity filtering. Neither alone is sufficient.
- **Bigram support** captures multi-word legislative concepts.
- **No impact on clustering.** UMAP + HDBSCAN operate on pre-computed embeddings. Topic *assignments* are unchanged; only *labels* improve.

### Trade-offs

- **Custom stopword list requires curation.** If the corpus changes (e.g., adding Missouri or Oklahoma bills), the legislative stopwords may need updating. Mitigated by keeping the list small (18 terms) and letting `max_df` handle the rest.
- **`max_df=0.85` may filter legitimate high-frequency terms.** In a corpus dominated by one policy area, that area's terms could exceed 85%. Unlikely for a full-biennium Kansas corpus (~800-1300 bills across all policy areas).

### Files changed

- `analysis/20_bill_text/bill_text.py` — `LEGISLATIVE_STOPWORDS` constant, `VECTORIZER_MAX_DF` constant, updated `CountVectorizer` with combined stopwords + `min_df`/`max_df`
- `analysis/design/bill_text.md` — updated vectorizer parameters in design table
- `docs/adr/0084-bill-text-analysis-phase-18.md` — updated vectorizer rows in settings table
- `docs/bill-text-nlp-deep-dive.md` — noted stopword requirement for c-TF-IDF quality
