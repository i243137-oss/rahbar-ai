# Corpus

Source: the public FAST-NUCES website. The indexing run processed **nine documents**: seven PDFs totaling **279 PDF pages** and two HTML pages. The source URLs and SHA-256 checksums are recorded in [corpus/MANIFEST.md](../corpus/MANIFEST.md). The source files are fetched into the ignored `corpus/raw/` directory.

| # | Required property | Document(s) | Evidence |
| --- | --- | --- | --- |
| 1 | At least 6 documents and 40 pages | All nine manifest entries | PDF page counts from the indexing logs: August 2023 rules (26), June 2026 rules (58), academic rules prospectus (5), student handbook (64), 2019–20 annual report (33), 2023–24 annual report (85), faculty start-up policy (8). Fee Structure and Academic Grading are HTML. |
| 2 | Two versions of the same policy | Undergraduate Academic Rules, August 2023 and June 2026 | `2023-08` is marked `superseded: true`; `2026-06` is marked `superseded: false`. The withdrawal deadline changed from the last day of classes to the last day of the makeup week. See golden questions q09 and q10. |
| 3 | Scanned PDF | Faculty Entrepreneurship and Technical Start-ups Policy; August 2023 rules | All 8 faculty policy pages went through the OCR fallback, and the index contains recovered text from that PDF. The August 2023 rules used OCR on 23 of 26 pages, including pages needed for historical policy retrieval. These are the observed OCR counts. |
| 4 | A table with real information | Fee Structure, Academic Grading, August 2023 rules | The tuition and refund tables preserve label/value pairs in HTML. The scanned grading table shows the harder case: several rows merge and two standalone labels are lost. See [PIPELINE.md](PIPELINE.md). |
| 5 | Long definitions | June 2026 Student Handbook | The generative AI policy contains extended guidance and qualifications in “4. Acceptable Uses of Generative AI Tools,” used by q06. |
| 6 | Deliberately absent topic | Hostel operating and application details | The corpus mentions hostel availability but omits the nightly curfew and application deadline asked about in q13 and q14. These are realistic questions for prospective residents. The separate transport and hostel facilities prospectus is outside the manifest. |
| 7 | Exact-term content | June 2026 Academic Rules Handbook, Academic Grading, Fee Structure | Clause `1.21` and the grade code `FA` are tested by q11 and q12. The corpus also includes CGPA thresholds and exact fee amounts. |

From the repository root, run `make fetch` to fetch the public corpus. Run `python corpus/fetch.py --verify` in the project environment to compare local files with the manifest.
