# ADR-0093: BERTopic Stopword Filtering for c-TF-IDF Topic Labels

**Date:** 2026-03-04
**Status:** Accepted

## Context

Phase 20 (Bill Text Analysis) uses BERTopic for topic modeling. BERTopic's c-TF-IDF layer extracts topic representations from a `CountVectorizer` internally. The default `CountVectorizer` has `stop_words=None`, meaning English function words (articles, prepositions, conjunctions) are included in the vocabulary.

This produced uninterpretable topic labels like "Topic 0: the, of, and, or, to" — pure stopwords that convey no policy content. The clustering itself (UMAP + HDBSCAN on semantic embeddings) was unaffected — only the c-TF-IDF label extraction was broken.

Phase 15 (Prediction) already uses `stop_words="english"` on its `TfidfVectorizer` for NMF topic modeling on short titles. Phase 20 was missing the equivalent configuration.

## Decision

Pass an explicit `CountVectorizer` to BERTopic with two settings:

1. **`stop_words="english"`** — scikit-learn's built-in English stopword list (318 words). Filters articles, prepositions, conjunctions, and other function words from the c-TF-IDF vocabulary.

2. **`ngram_range=(1, 2)`** — allows bigrams alongside unigrams. Legislative terminology is often multi-word ("tax credit", "school district", "criminal penalty", "motor vehicle"). Unigrams alone lose these compound terms.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    nr_topics="auto",
    calculate_probabilities=True,
    verbose=True,
)
```

### Why not a custom legislative stopword list?

Considered extending the English stopwords with legislative boilerplate terms ("bill", "act", "section", "shall", "amendment", "kansas"). Decided against it:

- The standard English list solves the immediate problem (function words dominating labels).
- Legislative terms like "bill" and "act" *are* meaningful when they appear in topic labels — they distinguish procedural from substantive topics.
- A custom list would require ongoing curation and introduces a maintenance burden without clear benefit.
- The existing text preprocessing in `bill_text_data.py` already strips boilerplate (enacting clauses, severability, effective dates, K.S.A. references) before embedding — the remaining legislative terms are genuinely topical.

## Consequences

### Benefits

- **Topic labels are now interpretable.** Instead of "the, of, and, or, to", topics show substantive terms like "education, school, curriculum" or "tax, credit, income".
- **Bigram support** captures multi-word legislative concepts that unigrams miss.
- **Consistent with Phase 15** (Prediction), which already uses `stop_words="english"`.
- **No impact on clustering.** UMAP + HDBSCAN operate on pre-computed semantic embeddings, not on the c-TF-IDF vocabulary. Topic *assignments* are unchanged; only topic *labels* improve.

### Trade-offs

- **Downstream phases must re-run.** Phase 21 (TBIP) and Phase 22 (Issue IRT) consume Phase 20 topic assignments. Since assignments are unchanged (only labels differ), re-running is optional but recommended for consistent labeling in reports.
- **Bigram vocabulary is larger.** The `(1, 2)` n-gram range increases the c-TF-IDF vocabulary size. This has negligible performance impact for a ~800-bill corpus.

### Files changed

- `analysis/20_bill_text/bill_text.py` — added `CountVectorizer` import, created `vectorizer_model`, passed to `BERTopic()`
- `analysis/design/bill_text.md` — added vectorizer parameters to design table
- `docs/adr/0084-bill-text-analysis-phase-18.md` — added vectorizer rows to settings table
- `docs/bill-text-nlp-deep-dive.md` — noted stopword requirement for c-TF-IDF quality
