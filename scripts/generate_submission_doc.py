#!/usr/bin/env python3
"""
Generate editable submission documents with clickable links and ONLY submission deliverables:
- submission.docx (Microsoft Word / Google Docs editable format)
- submission.md (Markdown format)
- submission.pdf (Single-page clean PDF format with clickable PDF link annotations)

Constraints:
- Only submission deliverables (Section 8 of Assignment)
- Times New Roman font (14pt for headings, 12pt for body)
- No coloring, no styling (monochrome black and white, standard layout)
- Clickable hyperlinks for all URLs
- Team: Umair Hassan (24i-3137) and Ahmad Tanveer (23i-3019)
"""

import os
import sys
import zipfile
import io

def generate_markdown():
    md_content = """# FAST-NUCES Islamabad -- CS4032 Software Construction & Development
## Assignment 02: RAHBAR -- Grounded University RAG on Kubernetes
### Submission Deliverables

**Team Members:**
- Umair Hassan (Roll No: 24i-3137)
- Ahmad Tanveer (Roll No: 23i-3019)

**Repository & Branch Details:**
- **Repository:** [https://github.com/i243137-oss/rahbar-ai](https://github.com/i243137-oss/rahbar-ai)
- **Main Branch Commit SHA (Production):** `5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b`
- **Dev Branch Commit SHA (Development):** `636e9dd703bf3e7a4a9558a8a8f21adf166ba447`

---

### Deliverable 1: GitHub Repository URL
- [https://github.com/i243137-oss/rahbar-ai](https://github.com/i243137-oss/rahbar-ai)

### Deliverable 2: Successful Continuous Delivery (cd.yml) Pipeline Run
- [https://github.com/i243137-oss/rahbar-ai/actions/runs/43048680250](https://github.com/i243137-oss/rahbar-ai/actions/runs/43048680250)
- **Workflow Status:** 100% Passed (3/3 jobs: Test Suite Verification, Build & Push Immutable SHA Images to Docker Hub, Ephemeral KinD Cluster Deploy & Rollout Verification).

### Deliverable 3: Docker Hub Images with Immutable SHA Tags
- **Docker Hub Profile:** [https://hub.docker.com/u/itxumairhassan](https://hub.docker.com/u/itxumairhassan)
- **API Image:** `docker.io/itxumairhassan/rahbar-api:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b`
- **Indexer Image:** `docker.io/itxumairhassan/rahbar-indexer:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b`
- **UI Image:** `docker.io/itxumairhassan/rahbar-ui:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b`

### Deliverable 4: Evaluation Results Committed in Repository
- **eval/results.json:** [https://github.com/i243137-oss/rahbar-ai/blob/main/eval/results.json](https://github.com/i243137-oss/rahbar-ai/blob/main/eval/results.json)
- **eval/baseline.json:** [https://github.com/i243137-oss/rahbar-ai/blob/main/eval/baseline.json](https://github.com/i243137-oss/rahbar-ai/blob/main/eval/baseline.json)
- **Verification Summary:**
  - Recall@5: 1.0 (100%)
  - Citation Validity: 1.0 (100% of generated citations match retrieved passages)
  - Abstention Correctness: 100% (honest refusal on all missing and out-of-scope queries)
  - Submission Checker (`check_submission.py`): 165 passed, 0 failures
"""
    with open("submission.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Generated submission.md")


def generate_docx():
    rels = {
        "rId1": "https://github.com/i243137-oss/rahbar-ai",
        "rId2": "https://github.com/i243137-oss/rahbar-ai/actions/runs/43048680250",
        "rId3": "https://hub.docker.com/u/itxumairhassan",
        "rId4": "https://github.com/i243137-oss/rahbar-ai/blob/main/eval/results.json",
        "rId5": "https://github.com/i243137-oss/rahbar-ai/blob/main/eval/baseline.json"
    }

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels_items = [
        f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="{url}" TargetMode="External"/>'
        for rid, url in rels.items()
    ]
    doc_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(doc_rels_items)}
</Relationships>"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>
        <w:sz w:val="24"/>
        <w:szCs w:val="24"/>
        <w:color w:val="000000"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
</w:styles>"""

    def p_heading(text):
        return f"""
    <w:p>
      <w:pPr><w:spacing w:before="240" w:after="80"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="28"/></w:rPr>
        <w:t>{text}</w:t>
      </w:r>
    </w:p>"""

    def p_subheading(text):
        return f"""
    <w:p>
      <w:pPr><w:spacing w:before="160" w:after="60"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="24"/></w:rPr>
        <w:t>{text}</w:t>
      </w:r>
    </w:p>"""

    def p_body(text, bold_prefix=""):
        bp = f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="24"/></w:rPr><w:t xml:space="preserve">{bold_prefix} </w:t></w:r>' if bold_prefix else ''
        return f"""
    <w:p>
      <w:pPr><w:spacing w:before="40" w:after="60"/></w:pPr>
      {bp}
      <w:r>
        <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
        <w:t>{text}</w:t>
      </w:r>
    </w:p>"""

    def p_link(label, url, rel_id):
        return f"""
    <w:p>
      <w:pPr><w:ind w:left="360"/><w:spacing w:before="40" w:after="60"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
        <w:t xml:space="preserve">{label} </w:t>
      </w:r>
      <w:hyperlink r:id="{rel_id}">
        <w:r>
          <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/><w:u w:val="single"/></w:rPr>
          <w:t>{url}</w:t>
        </w:r>
      </w:hyperlink>
    </w:p>"""

    def p_bullet(text):
        return f"""
    <w:p>
      <w:pPr><w:ind w:left="720"/><w:spacing w:before="40" w:after="40"/></w:pPr>
      <w:r>
        <w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>
        <w:t xml:space="preserve">*  {text}</w:t>
      </w:r>
    </w:p>"""

    body_xml = [
        p_heading("FAST-NUCES Islamabad -- CS4032 Software Construction &amp; Development"),
        p_heading("Assignment 02: RAHBAR -- Grounded University RAG on Kubernetes"),
        p_subheading("Assignment Submission Deliverables"),
        p_body("Umair Hassan (Roll No: 24i-3137)", bold_prefix="Team Member 1:"),
        p_body("Ahmad Tanveer (Roll No: 23i-3019)", bold_prefix="Team Member 2:"),
        p_body("5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b", bold_prefix="Main Branch Commit SHA (Production):"),
        p_body("636e9dd703bf3e7a4a9558a8a8f21adf166ba447", bold_prefix="Dev Branch Commit SHA (Development):"),
        
        p_subheading("Deliverable 1: GitHub Repository URL"),
        p_link("Repository URL:", "https://github.com/i243137-oss/rahbar-ai", "rId1"),
        
        p_subheading("Deliverable 2: Successful Continuous Delivery (cd.yml) Pipeline Run"),
        p_link("CD Workflow Run URL:", "https://github.com/i243137-oss/rahbar-ai/actions/runs/43048680250", "rId2"),
        p_bullet("Workflow Status: 100% Passed (3/3 jobs: Test Suite Verification, Build &amp; Push Immutable SHA Images, KinD Ephemeral Deployment &amp; Smoke Tests)."),
        
        p_subheading("Deliverable 3: Docker Hub Images with Immutable SHA Tags"),
        p_link("Docker Hub Registry Profile:", "https://hub.docker.com/u/itxumairhassan", "rId3"),
        p_bullet("API Service: docker.io/itxumairhassan/rahbar-api:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b"),
        p_bullet("Batch Indexer: docker.io/itxumairhassan/rahbar-indexer:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b"),
        p_bullet("Web Chat UI: docker.io/itxumairhassan/rahbar-ui:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b"),
        p_bullet("Deployment Standard: Pinned strictly by immutable commit SHA; mutable :latest tag forbidden."),
        
        p_subheading("Deliverable 4: Evaluation Artifacts Committed in Repository"),
        p_link("eval/results.json:", "https://github.com/i243137-oss/rahbar-ai/blob/main/eval/results.json", "rId4"),
        p_link("eval/baseline.json:", "https://github.com/i243137-oss/rahbar-ai/blob/main/eval/baseline.json", "rId5"),
        p_bullet("Recall@5: 1.0 (100% recall across golden test suite)"),
        p_bullet("Citation Validity: 1.0 (100% of claims cite verified retrieved passages)"),
        p_bullet("Abstention Correctness: 100% (honest refusal on all missing/out-of-scope questions)"),
        p_bullet("Submission Checker (scripts/check_submission.py): 165 passed, 0 failures")
    ]

    doc_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    {''.join(body_xml)}
    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    with zipfile.ZipFile("submission.docx", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/styles.xml", styles_xml)
        z.writestr("word/document.xml", doc_xml)
        
    print("Generated editable submission.docx with clickable links!")


def generate_single_page_pdf():
    # Also generate a clean 1-page deliverable-only PDF with clickable links
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import generate_submission_pdf as gen_pdf
    
    pdf = gen_pdf.AcademicPDFReport("submission.pdf")
    pdf.add_heading("FAST-NUCES Islamabad -- CS4032 Software Construction & Development", is_title=True)
    pdf.add_heading("Assignment 02: RAHBAR -- Grounded University RAG on Kubernetes")
    pdf.add_subheading("Assignment Submission Deliverables")
    pdf.add_divider_line()
    
    pdf.add_paragraph("Team Member 1: Umair Hassan (Roll No: 24i-3137)")
    pdf.add_paragraph("Team Member 2: Ahmad Tanveer (Roll No: 23i-3019)")
    pdf.add_paragraph("Main Branch Commit SHA (Production): 5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b")
    pdf.add_paragraph("Dev Branch Commit SHA (Development): 636e9dd703bf3e7a4a9558a8a8f21adf166ba447")
    pdf.add_divider_line()
    
    pdf.add_heading("Deliverable 1: GitHub Repository URL")
    pdf.add_bullet("https://github.com/i243137-oss/rahbar-ai")
    
    pdf.add_heading("Deliverable 2: Successful Continuous Delivery (cd.yml) Pipeline Run")
    pdf.add_bullet("Workflow Run URL: https://github.com/i243137-oss/rahbar-ai/actions/runs/43048680250")
    pdf.add_bullet("Status: 100% Passed (Test Verification, Docker Hub Build & Push, KinD Deploy & Rollout Smoke Tests).")
    
    pdf.add_heading("Deliverable 3: Docker Hub Images with Immutable SHA Tags")
    pdf.add_bullet("Registry: https://hub.docker.com/u/itxumairhassan")
    pdf.add_bullet("API Image: docker.io/itxumairhassan/rahbar-api:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b")
    pdf.add_bullet("Indexer Image: docker.io/itxumairhassan/rahbar-indexer:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b")
    pdf.add_bullet("UI Image: docker.io/itxumairhassan/rahbar-ui:5a614c8da1ffe94a422cdb9f21a19caea3ad8a3b")
    
    pdf.add_heading("Deliverable 4: Evaluation Results Committed in Repository")
    pdf.add_bullet("Results JSON: https://github.com/i243137-oss/rahbar-ai/blob/main/eval/results.json")
    pdf.add_bullet("Baseline JSON: https://github.com/i243137-oss/rahbar-ai/blob/main/eval/baseline.json")
    pdf.add_bullet("Metrics: Recall@5 = 1.0 (100%), Citation Validity = 1.0 (100%), Abstention Correctness = 100%.")
    pdf.add_bullet("Submission Checker (check_submission.py): 165 passed, 0 failures.")
    
    pdf.compile()
    print("Generated deliverable-only submission.pdf")

if __name__ == "__main__":
    generate_markdown()
    generate_docx()
    generate_single_page_pdf()
