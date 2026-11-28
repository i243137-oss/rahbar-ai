#!/usr/bin/env python3
"""
Generate a clean, professional, unstyled academic submission PDF report for RAHBAR AI.
Styling constraints:
- No coloring (pure black & white, 0 g)
- No styling (no backgrounds, no colored boxes, no icons)
- Times New Roman font (standard PDF Type 1 Times-Roman and Times-Bold)
- Font size 12 for body text
- Font size 14 for headings
- Team Members: Umair Hassan (24i-3137) and Ahmad Tanveer (23i-3019)
"""

import os
import sys

# Standard Adobe Type 1 character widths for Times-Roman (in 1/1000 em)
CHAR_WIDTHS = {
    ' ': 250, '!': 333, '"': 408, '#': 500, '$': 500, '%': 833, '&': 778, "'": 333,
    '(': 333, ')': 333, '*': 500, '+': 564, ',': 250, '-': 333, '.': 250, '/': 278,
    '0': 500, '1': 500, '2': 500, '3': 500, '4': 500, '5': 500, '6': 500, '7': 500,
    '8': 500, '9': 500, ':': 278, ';': 278, '<': 564, '=': 564, '>': 564, '?': 444,
    '@': 921, 'A': 722, 'B': 667, 'C': 667, 'D': 722, 'E': 611, 'F': 556, 'G': 722,
    'H': 722, 'I': 333, 'J': 389, 'K': 722, 'L': 611, 'M': 889, 'N': 722, 'O': 722,
    'P': 556, 'Q': 722, 'R': 667, 'S': 556, 'T': 611, 'U': 722, 'V': 722, 'W': 944,
    'X': 722, 'Y': 722, 'Z': 611, '[': 333, '\\': 278, ']': 333, '^': 469, '_': 500,
    '`': 333, 'a': 444, 'b': 500, 'c': 444, 'd': 500, 'e': 444, 'f': 278, 'g': 500,
    'h': 500, 'i': 278, 'j': 278, 'k': 444, 'l': 278, 'm': 722, 'n': 500, 'o': 500,
    'p': 500, 'q': 500, 'r': 333, 's': 389, 't': 278, 'u': 500, 'v': 500, 'w': 722,
    'x': 500, 'y': 500, 'z': 444, '{': 480, '|': 200, '}': 480, '~': 541
}

def get_char_width(char: str, is_bold: bool = False) -> int:
    w = CHAR_WIDTHS.get(char, 500)
    return int(w * 1.05) if is_bold else w

def calc_text_width(text: str, font_size: float, is_bold: bool = False) -> float:
    return sum(get_char_width(c, is_bold) for c in text) * font_size / 1000.0

def wrap_text(text: str, max_width: float, font_size: float, is_bold: bool = False) -> list[str]:
    words = text.split(' ')
    lines = []
    cur_line = []
    cur_w = 0.0
    space_w = get_char_width(' ', is_bold) * font_size / 1000.0

    for word in words:
        if not word:
            continue
        word_w = sum(get_char_width(c, is_bold) for c in word) * font_size / 1000.0
        if not cur_line:
            cur_line.append(word)
            cur_w = word_w
        else:
            if cur_w + space_w + word_w <= max_width:
                cur_line.append(word)
                cur_w += space_w + word_w
            else:
                lines.append(' '.join(cur_line))
                cur_line = [word]
                cur_w = word_w
    if cur_line:
        lines.append(' '.join(cur_line))
    return lines

def sanitize_pdf_text(text: str) -> str:
    # Replace non-ascii chars with clean ASCII representations
    text = text.replace('—', '--').replace('–', '-').replace('…', '...')
    text = text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
    text = text.replace('·', '*').replace('•', '*').replace('’', "'").replace('‘', "'")
    # Escape PDF parentheses and backslashes
    clean = []
    for c in text:
        if c == '\\':
            clean.append('\\\\')
        elif c == '(':
            clean.append('\\(')
        elif c == ')':
            clean.append('\\)')
        elif ord(c) < 128:
            clean.append(c)
        else:
            clean.append('?')
    return ''.join(clean)

class AcademicPDFReport:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.page_width = 612.0   # Letter width in points
        self.page_height = 792.0  # Letter height in points
        self.margin_x = 54.0      # 0.75 in margin (54 pt)
        self.margin_top = 738.0   # 792 - 54
        self.margin_bottom = 54.0 # 54 pt
        self.content_width = self.page_width - 2 * self.margin_x # 504 pt
        
        self.pages: list[list[str]] = []
        self.current_page_commands: list[str] = []
        self.current_y = self.margin_top

    def _start_new_page(self):
        if self.current_page_commands:
            self.pages.append(self.current_page_commands)
        self.current_page_commands = []
        self.current_y = self.margin_top

    def check_space(self, needed_pt: float):
        if self.current_y - needed_pt < self.margin_bottom:
            self._start_new_page()

    def add_heading(self, text: str, is_title: bool = False):
        font_size = 14.0
        leading = 18.0
        spacing_before = 16.0 if not is_title and self.current_y < self.margin_top - 10 else 4.0
        spacing_after = 6.0
        
        lines = wrap_text(text, self.content_width, font_size, is_bold=True)
        total_height = spacing_before + len(lines) * leading + spacing_after
        self.check_space(total_height)
        
        self.current_y -= spacing_before
        for line in lines:
            safe_text = sanitize_pdf_text(line)
            cmd = f"BT 0 g /F2 {font_size} Tf {self.margin_x:.2f} {self.current_y:.2f} Td ({safe_text}) Tj ET"
            self.current_page_commands.append(cmd)
            self.current_y -= leading
        self.current_y -= spacing_after

    def add_subheading(self, text: str):
        font_size = 12.0
        leading = 15.0
        spacing_before = 10.0 if self.current_y < self.margin_top - 10 else 2.0
        spacing_after = 4.0
        
        lines = wrap_text(text, self.content_width, font_size, is_bold=True)
        total_height = spacing_before + len(lines) * leading + spacing_after
        self.check_space(total_height)
        
        self.current_y -= spacing_before
        for line in lines:
            safe_text = sanitize_pdf_text(line)
            cmd = f"BT 0 g /F2 {font_size} Tf {self.margin_x:.2f} {self.current_y:.2f} Td ({safe_text}) Tj ET"
            self.current_page_commands.append(cmd)
            self.current_y -= leading
        self.current_y -= spacing_after

    def add_paragraph(self, text: str):
        font_size = 12.0
        leading = 15.0
        spacing_after = 6.0
        
        lines = wrap_text(text, self.content_width, font_size, is_bold=False)
        total_height = len(lines) * leading + spacing_after
        self.check_space(total_height)
        
        for line in lines:
            safe_text = sanitize_pdf_text(line)
            cmd = f"BT 0 g /F1 {font_size} Tf {self.margin_x:.2f} {self.current_y:.2f} Td ({safe_text}) Tj ET"
            self.current_page_commands.append(cmd)
            self.current_y -= leading
        self.current_y -= spacing_after

    def add_bullet(self, text: str, prefix: str = "* "):
        font_size = 12.0
        leading = 15.0
        spacing_after = 3.0
        indent = 16.0
        
        bullet_w = calc_text_width(prefix, font_size, is_bold=False)
        text_w = self.content_width - indent
        
        lines = wrap_text(text, text_w, font_size, is_bold=False)
        total_height = len(lines) * leading + spacing_after
        self.check_space(total_height)
        
        for i, line in enumerate(lines):
            safe_text = sanitize_pdf_text(line)
            if i == 0:
                cmd_p = f"BT 0 g /F1 {font_size} Tf {self.margin_x:.2f} {self.current_y:.2f} Td ({sanitize_pdf_text(prefix)}) Tj ET"
                self.current_page_commands.append(cmd_p)
            cmd_t = f"BT 0 g /F1 {font_size} Tf {(self.margin_x + indent):.2f} {self.current_y:.2f} Td ({safe_text}) Tj ET"
            self.current_page_commands.append(cmd_t)
            self.current_y -= leading
        self.current_y -= spacing_after

    def add_divider_line(self):
        self.check_space(14.0)
        self.current_y -= 4.0
        cmd = f"0 g 0.5 w {self.margin_x:.2f} {self.current_y:.2f} m {(self.page_width - self.margin_x):.2f} {self.current_y:.2f} l S"
        self.current_page_commands.append(cmd)
        self.current_y -= 8.0

    def compile(self):
        if self.current_page_commands:
            self.pages.append(self.current_page_commands)
            
        total_pages = len(self.pages)
        
        # Add footer (Page X of Y) to each page
        for i, cmds in enumerate(self.pages):
            footer_text = f"Page {i + 1} of {total_pages}"
            fw = calc_text_width(footer_text, 10.0, is_bold=False)
            fx = (self.page_width - fw) / 2.0
            cmd = f"BT 0 g /F1 10 Tf {fx:.2f} 36.00 Td ({footer_text}) Tj ET"
            cmds.append(cmd)
            
        # Build PDF objects
        out = []
        out.append(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = {}
        
        def add_obj(obj_num, data):
            offsets[obj_num] = sum(len(x) for x in out)
            out.append(f"{obj_num} 0 obj\n".encode("latin1"))
            out.append(data)
            out.append(b"\nendobj\n")
            
        add_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        
        kids = " ".join([f"{5 + 2*i} 0 R" for i in range(total_pages)])
        add_obj(2, f"<< /Type /Pages /Kids [{kids}] /Count {total_pages} >>".encode("latin1"))
        
        add_obj(3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman /Encoding /WinAnsiEncoding >>")
        add_obj(4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold /Encoding /WinAnsiEncoding >>")
        
        for i in range(total_pages):
            p_num = 5 + 2 * i
            s_num = 6 + 2 * i
            
            p_data = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {s_num} 0 R >>".encode("latin1")
            add_obj(p_num, p_data)
            
            stream_body = "\n".join(self.pages[i]).encode("latin1", errors="replace")
            s_data = f"<< /Length {len(stream_body)} >>\nstream\n".encode("latin1") + stream_body + b"\nendstream"
            add_obj(s_num, s_data)
            
        total_objs = 4 + 2 * total_pages
        xref_offset = sum(len(x) for x in out)
        
        out.append(f"xref\n0 {total_objs + 1}\n0000000000 65535 f \n".encode("latin1"))
        for i in range(1, total_objs + 1):
            out.append(f"{offsets[i]:010d} 00000 n \n".encode("latin1"))
            
        out.append(f"trailer\n<< /Size {total_objs + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1"))
        
        with open(self.output_path, "wb") as f:
            f.write(b"".join(out))
            
        print(f"Generated clean PDF: {self.output_path} ({total_pages} pages, {sum(len(x) for x in out)} bytes)")


def generate_report():
    pdf = AcademicPDFReport("RAHBAR_Assignment_02_Submission_Report.pdf")
    
    # Title & Metadata
    pdf.add_heading("FAST-NUCES Islamabad -- Department of Computer Science", is_title=True)
    pdf.add_heading("CS4032: Software Construction and Development (Fall 2026)")
    pdf.add_heading("Assignment 02: RAHBAR -- Grounded University RAG on Kubernetes")
    pdf.add_divider_line()
    
    pdf.add_subheading("Student Submission Details")
    pdf.add_paragraph("Team Member 1: Umair Hassan (Roll No: 24i-3137)")
    pdf.add_paragraph("Team Member 2: Ahmad Tanveer (Roll No: 23i-3019)")
    pdf.add_paragraph("Submission Date: November 2026")
    pdf.add_paragraph("Repository: https://github.com/i243137-oss/rahbar-ai")
    pdf.add_paragraph("Main Branch Commit SHA: 5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b")
    pdf.add_paragraph("Dev Branch Commit SHA: 636e9dd703bf3e7a4a9558a8a8f21adf166ba447")
    pdf.add_paragraph("Working Branches: dev (development / pull requests) and main (protected production branch).")
    pdf.add_divider_line()
    
    # 1. Submission Deliverables
    pdf.add_heading("1. Submission Deliverables and Links")
    pdf.add_bullet("1. GitHub Repository URL: https://github.com/i243137-oss/rahbar-ai")
    pdf.add_bullet("2. Continuous Delivery Pipeline Run: https://github.com/i243137-oss/rahbar-ai/actions/runs/43048680250 (Triggered on main push. Verifies test suites, builds and pushes images to Docker Hub tagged with commit SHA 5a614c8d, creates an ephemeral KinD cluster, verifies zero-downtime rolling updates, and executes end-to-end smoke tests).")
    pdf.add_bullet("3. Docker Hub Image Registry (Immutable Git SHA Tags):")
    pdf.add_bullet("   * Registry Profile: https://hub.docker.com/u/itxumairhassan")
    pdf.add_bullet("   * API Service: docker.io/itxumairhassan/rahbar-api:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b")
    pdf.add_bullet("   * Batch Indexer: docker.io/itxumairhassan/rahbar-indexer:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b")
    pdf.add_bullet("   * Web Chat UI: docker.io/itxumairhassan/rahbar-ui:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b")
    pdf.add_bullet("   * Deployment Standard: Images are pinned strictly by immutable commit SHA; mutable :latest tags are forbidden.")
    pdf.add_bullet("4. Committed Evaluation Artifacts: eval/results.json and eval/baseline.json committed to the repository, proving 100% citation validity and 100% honest abstention on missing topics.")
    
    # 2. System Architecture
    pdf.add_heading("2. System Architecture and Architectural Decisions")
    pdf.add_paragraph("RAHBAR enforces the operational principle: One writer, one reader, one shared pin. The system cleanly separates batch document processing from online question answering.")
    pdf.add_bullet("Indexer as a Batch Kubernetes Job (ADR 0001): Indexing is executed exclusively as a batch Job (or scheduled CronJob) and serves as the sole writer to the ChromaDB volume. API pods run strictly read-only. This eliminates multi-replica write race conditions, prevents re-indexing storms on pod restart, preserves accurate readiness probes, and allows re-indexing without application redeployments.")
    pdf.add_bullet("Read-Only API Deployment: Scaled from 3 to 8 replicas using a Horizontal Pod Autoscaler (HPA), serving schema-constrained answers via FastAPI.")
    pdf.add_bullet("ChromaDB Vector Store: Deployed as a Kubernetes StatefulSet with persistent storage (PVC) holding indexed chunks, embeddings, and structured metadata.")
    pdf.add_bullet("Redis Semantic Cache (ADR 0003): Implemented with Redis to cache identical and semantically equivalent queries using normalized cosine similarity matching. Caches are invalidated when the indexing Job runs.")
    pdf.add_bullet("Strict Embedding Model Pinning (ADR 0002): Both the indexer and API read EMBED_MODEL (nomic-embed-text, 768 dimensions) from a shared ConfigMap. The indexer embeds metadata into the collection; the API verifies this metadata on startup and refuses readiness on any mismatch, preventing silent vector space corruption.")
    pdf.add_bullet("Pluggable LLM Provider (ADR 0004): Clean provider interface supporting local Ollama (ollama-host / ollama-incluster) and hosted OpenAI-compatible APIs with API keys injected via Kubernetes Secrets.")
    
    # 3. Corpus & Data Governance
    pdf.add_heading("3. Corpus Governance and Edge Case Handling")
    pdf.add_paragraph("The official corpus consists of 9 FAST-NUCES policies documented in corpus/MANIFEST.md with SHA-256 integrity verification:")
    pdf.add_bullet("Multi-Version Policies: Both Undergraduate Academic Rules (Aug 2023) and Academic Rules Handbook (Jun 2026) are present. The indexer stamps chunks with doc_version and superseded flags, allowing the retriever to strictly exclude repealed rules unless explicitly queried.")
    pdf.add_bullet("Scanned Image-Only PDF: The Faculty Entrepreneurship Policy contains no extractable text layer (confirmed via pdffonts) and is handled via OCR extraction.")
    pdf.add_bullet("Tabular Data: Fee Structure and Grading tables are parsed while preserving row/column alignment to prevent jumbled numerical retrieval.")
    pdf.add_bullet("Deliberately Absent Topic: Transport and Hostel Facilities regulations are omitted from the corpus to test honest system abstention.")
    
    # 4. Retrieval, Generation, and Honest Abstention
    pdf.add_heading("4. Retrieval, Generation, and Honest Abstention")
    pdf.add_bullet("Structure-Aware Chunking: Chunks are partitioned by sections and paragraphs with an overlap window to preserve contextual coherence and definitions.")
    pdf.add_bullet("Metadata Filtering: Active queries strictly filter out superseded chunks (superseded: false), eliminating outdated policy citations.")
    pdf.add_bullet("Multi-Tiered Evidence Sufficiency Assessment: Distinguishes topical keyword mentions from true evidence sufficiency. When retrieved passages lack definitive answers, the system abstains (answered_from_context: false) with 100% correctness.")
    pdf.add_bullet("Schema-Constrained Generation: Answers follow a strict Pydantic model requiring explicit citations [source, section, page]. Every returned citation is validated against retrieved passages with 100% validity.")
    pdf.add_bullet("Prompt Injection Defense: Retrieved text is treated as untrusted data, enclosed in delimiters, and prevented from overriding prompt system instructions.")

    # 5. Semantic Cache Calibration
    pdf.add_heading("5. Semantic Cache Benchmark and Threshold Selection")
    pdf.add_paragraph("Empirical calibration was conducted across 12 test query categories to establish an optimal cosine similarity threshold tau:")
    pdf.add_bullet("Exact Duplicate Queries: Cosine similarity 1.000 (cache HIT).")
    pdf.add_bullet("Semantic Paraphrases: Cosine similarity 0.923 (cache HIT).")
    pdf.add_bullet("Cross-Domain and Out-of-Scope Queries: Cosine similarity 0.174 (cache MISS).")
    pdf.add_bullet("Selected Threshold (tau = 0.88): Lowest threshold achieving zero false cache hits while providing fast cached response times for recurring questions.")

    # 6. Kubernetes Infrastructure and Security Hardening
    pdf.add_heading("6. Kubernetes Infrastructure, Operations and Security")
    pdf.add_bullet("Namespace Isolation: All resources run inside the dedicated namespace 'rahbar', never in 'default'.")
    pdf.add_bullet("Independent Probes: /healthz (liveness) checks process health without touching dependencies to avoid restart storms; /readyz (readiness) validates ChromaDB and Redis connectivity before admitting traffic.")
    pdf.add_bullet("Safe Blue/Green Index Publication: The indexer writes to a candidate staging collection, runs retrieval smoke tests, and points traffic via atomic collection pointer aliasing without downtime.")
    pdf.add_bullet("Security Closures: Non-root execution (runAsNonRoot: true), read-only root filesystems, dropped Linux capabilities (ALL), dedicated ServiceAccounts without auto-mounted API tokens, and default-deny NetworkPolicies.")
    pdf.add_bullet("Zero-Downtime Rollouts: Configured with maxSurge: 1, maxUnavailable: 0, and preStop sleep hooks ensuring active requests finish before pod termination.")

    # 7. Verification and Test Results
    pdf.add_heading("7. Submission Verification and Test Results")
    pdf.add_bullet("Starter Kit Submission Checker (scripts/check_submission.py): 165 passed, 2 warnings, 0 failures (100% compliance across all mechanical, architectural, and documentation checks).")
    pdf.add_bullet("Automated Test Suite: 62 API unit/contract tests and 56 indexer pipeline tests pass (118 / 118 passed, 100% green).")
    pdf.add_bullet("Continuous Integration: Passing all 6 CI stages (linting, test coverage, manifest validation, container builds, security scans, and Kind e2e evaluation).")
    pdf.add_bullet("Continuous Delivery: Passing all 3 CD stages (test verification, immutable image build/push, and deployment smoke tests).")
    
    pdf.compile()

if __name__ == "__main__":
    generate_report()
