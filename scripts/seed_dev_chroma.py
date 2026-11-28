#!/usr/bin/env python3
"""
Seed ChromaDB with university policies and test fixtures using deterministic 768-dim embeddings.
Can run locally or inside a Kubernetes pod.
"""

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

# Deterministic semantic embedding
def _token_vec(token: str, dim: int = 768) -> list[float]:
    values: list[float] = []
    seed = token.encode("utf-8")
    while len(values) < dim:
        seed = hashlib.sha256(seed).digest()
        values.extend((b - 127.5) / 127.5 for b in seed)
    values = values[:dim]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]

def embed_text(text: str, dim: int = 768) -> list[float]:
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    if not tokens:
        return [1.0 / math.sqrt(dim)] * dim
    combined = [0.0] * dim
    for tok in tokens:
        w = 2.0 if any(c.isdigit() for c in tok) else 1.0
        v = _token_vec(tok, dim)
        for i in range(dim):
            combined[i] += v[i] * w
    for i in range(len(text) - 3):
        ng = text[i:i+4].lower()
        v = _token_vec(ng, dim)
        for j in range(dim):
            combined[j] += v[j] * 0.2
    norm = math.sqrt(sum(v * v for v in combined)) or 1.0
    return [v / norm for v in combined]

DOCUMENTS = [
    {
        "id": "rules-2026-attendance",
        "text": "Margalla Institute of Technology: Academic Rules (2026 edition). Section 1. Attendance: A student must attend at least 80 percent of the lectures of each course to be allowed to sit the final examination. The Head of Department may condone a further absence only for documented medical reasons. A student who falls below the minimum attendance receives the grade FA in that course.",
        "source": "academic-rules-2026.md",
        "section": "1. Attendance",
        "page": 1,
        "doc_version": "2026",
        "superseded": False,
    },
    {
        "id": "rules-2026-warning",
        "text": "Margalla Institute of Technology: Academic Rules (2026 edition). Section 2. Academic warning: A student whose cumulative grade point average (CGPA) falls below 2.00 at the end of a regular semester is placed on academic warning. A student who remains on academic warning for three consecutive semesters is dropped from the programme.",
        "source": "academic-rules-2026.md",
        "section": "2. Academic warning",
        "page": 1,
        "doc_version": "2026",
        "superseded": False,
    },
    {
        "id": "rules-2026-repeat",
        "text": "Margalla Institute of Technology: Academic Rules (2026 edition). Section 3. Repeating a course: A student may repeat any course to improve the CGPA. Only the grade of the most recent attempt counts, even when it is lower than an earlier attempt.",
        "source": "academic-rules-2026.md",
        "section": "3. Repeating a course",
        "page": 1,
        "doc_version": "2026",
        "superseded": False,
    },
    {
        "id": "rules-2023-warning",
        "text": "Margalla Institute of Technology: Academic Rules (2023 edition). Section 2. Academic warning: A student whose cumulative grade point average (CGPA) falls below 1.80 at the end of a regular semester is placed on academic warning. This policy was repealed and superseded by the 2026 edition.",
        "source": "academic-rules-2023.md",
        "section": "2. Academic warning",
        "page": 1,
        "doc_version": "2023",
        "superseded": True,
    },
    {
        "id": "fees-2026-tuition",
        "text": "Margalla Institute of Technology: Fee Schedule (Fall 2026). Section 1. Tuition fees per credit hour: BS (all disciplines) tuition is PKR 10,000 per credit hour. MS tuition is PKR 12,500 per credit hour.",
        "source": "fees-2026.md",
        "section": "1. Tuition",
        "page": 1,
        "doc_version": "2026",
        "superseded": False,
    },
    {
        "id": "fees-2026-refunds",
        "text": "Margalla Institute of Technology: Fee Schedule (Fall 2026). Section 3. Refunds: Tuition is refunded in full when a course is dropped within the first 10 days of classes, and at 50 percent between day 11 and day 20. No refund is given after day 20.",
        "source": "fees-2026.md",
        "section": "3. Refunds",
        "page": 1,
        "doc_version": "2026",
        "superseded": False,
    },
    {
        "id": "fees-2026-charges",
        "text": "Margalla Institute of Technology: Fee Schedule (Fall 2026). Section 2. Other charges: Late registration fee is PKR 3,000. Official transcript copy fee is PKR 1,500. Campus transfer processing fee is PKR 5,000.",
        "source": "fees-2026.md",
        "section": "2. Other charges",
        "page": 1,
        "doc_version": "2026",
        "superseded": False,
    },
]

def seed_chroma(host="dev-chroma", port=8000, collection_name="rahbar"):
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    
    print(f"Connecting to Chroma at {host}:{port}...")
    client = chromadb.HttpClient(
        host=host,
        port=port,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    # Try deleting existing collection to start clean
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
        
    meta = {
        "embed_model": "dummy-768",
        "dimension": 768,
        "index_id": "seed-fixtures-2026",
        "hnsw:space": "cosine"
    }
    col = client.get_or_create_collection(name=collection_name, metadata=meta)
    
    ids = [d["id"] for d in DOCUMENTS]
    texts = [d["text"] for d in DOCUMENTS]
    embeddings = [embed_text(d["text"]) for d in DOCUMENTS]
    metadatas = [
        {
            "source": d["source"],
            "section": d["section"],
            "page": d["page"],
            "doc_version": d["doc_version"],
            "superseded": d["superseded"],
        }
        for d in DOCUMENTS
    ]
    
    col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print(f"Indexed {col.count()} documents into collection '{collection_name}'")
    
    # Also update pointer collection
    try:
        client.delete_collection(f"{collection_name}_pointer")
    except Exception:
        pass
    ptr = client.get_or_create_collection(
        name=f"{collection_name}_pointer",
        metadata={"active_collection": collection_name, "index_id": "seed-fixtures-2026"}
    )
    print(f"Pointer collection '{collection_name}_pointer' active -> '{collection_name}'")
    
    # Test retrieval
    test_q = "What CGPA puts a student on academic warning?"
    q_vec = embed_text(test_q)
    res = col.query(
        query_embeddings=[q_vec],
        n_results=3,
        where={"superseded": False},
        include=["documents", "metadatas", "distances"]
    )
    print("Verification query results:")
    for doc, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sim = 1.0 - dist
        print(f"  Score: {sim:.4f} | Source: {m['source']} ({m['section']}) | Text: {doc[:80]}...")

if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CHROMA_HOST", "dev-chroma")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("CHROMA_PORT", 8000))
    seed_chroma(host, port)
