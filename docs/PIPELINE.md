# Indexing and retrieval pipeline

<!-- STARTER-TODO(3.1, 3.3, 5.2): fill each section with your numbers, then delete this comment. -->

## Extraction
Which documents needed OCR, one before-and-after example of a table, what could not be recovered.

The HTML tuition and refund tables preserved their row labels and values.
For example, the tuition row became `BBA / BS | Rs. 12,000`, and the
surrounding text preserved that this is the rate per credit hour for 2026–27.

The August 2023 rules preserved the SGPA and CGPA explanation. In the
grading chunk tagged PDF page 9, the standalone S and U rows visible in
the source excerpt were lost. Clause numbers 2.4 through 2.11 were grouped
before their text, making individual clause references unreliable.
Some OCR noise also survived in the table of contents.

Both chunks tagged PDF page 39 of the June 2026 academic rules handbook
interleave the exam retake list with nearby notes. The original page is
needed to verify answers based on those passages.

### Section labels

The chunker sometimes treats cover text such as `ACADEMIC RULES AND` and
`NATIONAL UNIVERSITY` as section headings. Some numbered clauses remain
under broader section labels. For example, clause 1.21 is in the body
of a chunk labeled `1. Introduction`.

## Cleaning
The rules you applied, and one thing you deliberately did not strip.

## Chunking
Size and overlap you shipped, storage amplification (stored tokens / original tokens),
and the three configurations you measured:

| Size (tokens) | Overlap (tokens) | recall@5 | Chunks stored |
| --- | --- | --- | --- |

## Embedding
Model, vector dimension, where the pin lives, how the API checks it.

## Semantic cache
Thresholds you tried, the false hit that ruled one out, the one you shipped.

## Evaluation method
How recall@k matches retrieved passages to golden ones (by source and section? by chunk id?).
