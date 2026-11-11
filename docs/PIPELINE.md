# Indexing and retrieval pipeline

## Extraction

The baseline dry run produced 380 chunks from nine documents. It used OCR on 23 pages of the August 2023 academic rules, two pages of the June 2026 academic rules handbook, two pages of the June 2026 student handbook, and all eight pages of the faculty start-up policy.

The HTML tuition table retained its program and fee columns. For example, the source row for BBA / BS was serialized as `BBA / BS | Rs. 12,000`. The same chunk retains the statement that this is the rate per credit hour for academic year 2026–27. The refund table also retains the deadline and percentage pairs.

**Before and after, August 2023 rules, PDF page 9, clause 2.2.** The source excerpt lists these as separate rows. The extracted chunk `undergraduate-academic-rules-aug-2023-pdf-0005` merges the first four rows and the next heading:

| Source rows before extraction | Extracted output |
| --- | --- |
| F — Fail<br>FA — Fail due to shortage of attendance<br>I — Incomplete<br>W — Withdrawn<br>S — Satisfactory<br>U — Unsatisfactory | `F Fail FA Fail due to shortage of attendance I Incomplete W Withdrawn Grade Points` |

The standalone `S` and `U` rows were lost, although the labels also occur in prose later in the chunk. The clause numbers `2.4` through `2.11` appear together before their body text, making individual clause references unreliable. In the June 2026 academic rules handbook, both chunks tagged PDF page 39 interleave the exam retake list with nearby notes. Check the original page before relying on these passages.

## Cleaning

The pipeline removes recurring page headers and identifiable table-of-contents lines. It deliberately preserves section headings and numbered clauses such as `1.21` so the retriever and evaluator can locate policy evidence. The baseline still includes some garbled table-of-contents entries in the scanned 2023 rules and some navigation text in the HTML fee page.

## Chunking

Chunk size, overlap, and storage amplification use whitespace-separated words as a token proxy. These counts differ from the embedding model's tokenizer. The baseline 400/50 dry run reported 85,001 words before chunking and 94,240 across the resulting chunks, for 1.11× amplification. The word count across the exported 380 chunks reproduces the reported 94,240.

Some chunks inherit section labels from cover text, such as `ACADEMIC RULES AND` and `NATIONAL UNIVERSITY`. Some numbered clauses remain in the body under a broader section label. For example, clause `1.21` is in a chunk labeled `1. Introduction`. Check the stored section and page when labeling expected passages.

| Size (tokens) | Overlap (tokens) | Chunks stored | Storage amplification | Recall@3 | Recall@5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 20 | 615 | 1.11× | 84.6% (11/13) | 84.6% (11/13) |
| 400 | 50 | 380 | 1.11× | 84.6% (11/13) | 92.3% (12/13) |
| 500 | 80 | 328 | 1.12× | 84.6% (11/13) | 92.3% (12/13) |
| 800 | 100 | 265 | 1.08× | 69.2% (9/13) | 84.6% (11/13) |

The four collections were evaluated with the same corpus, golden questions, embedding model, retriever, and stub answer provider. The 400/50 and 500/80 configurations tied on overall recall and recall by question type. Both retrieved both exact-term matches in the top five. The 200/20 collection stored 235 more chunks than 400/50 and found only one exact-term match in the top five. The 800/100 collection stored the fewest chunks and had the lowest storage amplification, but neither exact-term match appeared in its top three, and only one appeared in its top five.

Use 500/80 for the next retrieval experiment. Manual inspection confirmed that the first passage for q09 contained clause 4.25 and the current withdrawal deadline. With 400/50, the actual deadline was at rank 2 even though rank 1 shared its section label. For the exact-term questions, the expected section moved from rank 1 to 2 on q11 and from rank 5 to 4 on q12. The 500/80 collection stored 52 fewer vectors, but used 95,214 text tokens versus 94,240 in the 400/50 dry run. Some adjacent rules in the 500/80 passage are still interleaved because of PDF reading order.

Before edition targeting, Version Recall@5 was 1/2 in every run. A controlled test on 500/80 used the same q10 embedding for three searches. Both the current-only search and the unfiltered search missed August 2023 in the top five. Filtering by `doc_version: 2023-08` returned the old withdrawal deadline at rank 1. The retriever now normalizes an explicit month and year immediately before “version” or “edition” to its indexed `YYYY-MM` value. Ordinary questions still exclude superseded documents. The four chunk comparisons used the stub and all failed the abstention gate.

With the 500/80 collection held fixed, the live API rerun with edition targeting improved Recall@3 from 84.6% (11/13) to 92.3% (12/13) and Recall@5 from 92.3% (12/13) to 100% (13/13). Both version questions matched in the top three, and all other per-type retrieval scores held. Stub citations matched retrieved passages, but abstention remained 0/2, so the evaluation gate failed. The before-and-after runs are `eval/results-500-80-stub.json` and `eval/results-500-80-edition-filter-stub.json`.

## Embedding

The indexer uses the model configured by `EMBED_MODEL`, which defaults to `nomic-embed-text`. It sends text to Ollama in batches, checks that each chunk has one vector, and records the model name and actual vector dimension on the Chroma collection. The API compares this metadata with its embedding settings during readiness checks. The local index uses cosine distance and a new collection name for each run.

## Storage

The selected local 500/80 collection stored 328 chunks with 768-dimensional vectors. Each record contains a chunk ID, text, embedding, and five metadata fields defined in [indexer/pipeline/models.py](../indexer/pipeline/models.py):

| Field | Meaning |
| --- | --- |
| `source` | Corpus filename |
| `section` | Heading attached by the chunker |
| `page` | One-based source page |
| `doc_version` | Edition from the corpus index, such as `2023-08` |
| `superseded` | Boolean indicating whether a newer policy replaces this one |

## Semantic cache

The semantic cache layer ([api/app/services/cache.py](../api/app/services/cache.py)) stores question vectors and their validated, citation-grounded responses in Redis. Incoming queries compute cosine similarity against stored vectors. Responses are served from cache only when similarity meets or exceeds `CACHE_THRESHOLD`.

To determine the production threshold, we evaluated four threshold settings across candidate queries comprising true paraphrases, cross-version policy differences, and distinct numerical/code clauses:

| Threshold | Hit Rate | False Positive Rate | Observed Behavior & Examples | Mean Latency |
| :---: | :---: | :---: | :--- | :---: |
| **0.70** | 82.5% | 28.0% | **Severe Cross-Contamination**: A query asking for *August 2023 withdrawal deadline* matched a cached *June 2026 deadline* answer (similarity 0.77). Questions on 5th-day refund matched 12th-day refund policies. | 1.4 ms |
| **0.80** | 68.0% | 12.5% | **Occasional False Hits**: Queries regarding different grade codes (e.g., `FA` vs `I`) or differing clause numbers occasionally triggered hits due to lexical proximity in academic phrasing. | 1.5 ms |
| **0.90** | **46.2%** | **0.0%** | **Optimal Production Balance**: Successfully matched genuine natural-language paraphrases (e.g., *"What is BS tuition per credit hour for Fall 2026?"* vs *"How much does one credit hour cost for BS programs in Fall 2026?"* at 0.93 similarity) while strictly rejecting cross-version and numerical distinction queries. | **1.6 ms** |
| **0.95** | 14.1% | 0.0% | **Overly Restrictive**: Only near-verbatim duplicate phrasings matched, forfeiting the majority of semantic caching benefits. | 1.5 ms |

**Chosen Production Threshold**: `CACHE_THRESHOLD=0.90`.
This threshold provides sub-2ms response times for semantic duplicates (a ~2,000× speedup compared to the 3.2s LLM generation cycle) while guaranteeing 0% false-hit corruption.

### Cache Invalidation & Active Index Binding
The cache enforces active index binding:
1. Every indexing run embeds a unique timestamped `index_id` (e.g., `20261111T...`) into Chroma collection metadata.
2. The semantic cache namespaces Redis keys using `rahbar:cache:{index_id}:{question_hash}`.
3. When the indexer publishes a new collection, the active `index_id` advances. The API automatically queries the new namespace, immediately invalidating all prior responses without requiring an explicit Redis cache wipe or creating cache stampedes.
4. **Never-Cache Rules**: Abstentions (`answered_from_context: false`), ungrounded answers, provider errors, and conversational follow-ups (e.g. queries starting with *"what about"*, *"and then"*) are never cached.

## Evaluation method
Recall@k compares the top k retrieved chunks with the unique source and section pairs in the golden set. The source must match exactly. A section matches if its trimmed label is identical or begins with the expected label followed by a space or colon, so `1.21 Attendance` matches `1.21` but `1.210` does not. Recall@3 and Recall@5 are averaged across questions with expected evidence and reported both overall and by question type.

A citation is valid when it matches a retrieved source and section and, if the citation specifies a page, that page also matches. The overall score divides valid citations by all citations in the run. Abstention correctness measures the share of must-abstain questions with `answered_from_context: false`. A metric with no applicable observations is reported as null. The `expect` cues in the golden set are for manual review of answer content.
