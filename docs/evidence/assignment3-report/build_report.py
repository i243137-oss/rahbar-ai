from pathlib import Path
import json
import re

import cairo
import gi
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo

source = Path("docs/ASSIGNMENT3-REPORT.md")
out = Path("docs/Assignment3-RAHBAR-Report.pdf")
shots = Path("docs/evidence/assignment3-report/screenshots")
pw, ph = 595.28, 841.89
margin = 49
width = pw - 2 * margin
navy = (0.08, 0.16, 0.26)
muted = (0.27, 0.34, 0.42)
green = (0.04, 0.42, 0.40)
pale = (0.95, 0.97, 0.98)

s = cairo.PDFSurface(str(out), pw, ph)
s.set_metadata(cairo.PDF_METADATA_TITLE, "RAHBAR AI · Assignment 03 RAGOps")
s.set_metadata(cairo.PDF_METADATA_AUTHOR, "Ahmed Tanweer and Umair Hassan")
c = cairo.Context(s)
page_count = 0

def text(value, x, y, w, size=10.5, color=navy, bold=False, draw=True):
    layout = PangoCairo.create_layout(c)
    PangoCairo.context_set_resolution(layout.get_context(), 72.0)
    font = Pango.FontDescription()
    font.set_family("Liberation Sans")
    font.set_absolute_size(size * Pango.SCALE)
    if bold:
        font.set_weight(Pango.Weight.BOLD)
    layout.set_font_description(font)
    layout.set_width(int(w * Pango.SCALE))
    layout.set_wrap(Pango.WrapMode.WORD_CHAR)
    layout.set_spacing(int(2 * Pango.SCALE))
    layout.set_text(value.replace("`", ""), -1)
    extents = layout.get_extents()[1]
    height = extents.height / Pango.SCALE
    if draw:
        c.set_source_rgb(*color)
        c.move_to(x, y)
        PangoCairo.show_layout(c, layout)
    return height

def box(x, y, w, h, color):
    c.set_source_rgb(*color)
    c.rectangle(x, y, w, h)
    c.fill()

def page(w, h, tag):
    global page_count
    page_count += 1
    s.set_size(w, h)
    box(0, 0, w, h, (1,1,1))
    box(margin, 37, 20, 3, green)
    text("RAHBAR AI  /  CS4032", margin+28, 32, w-150, 9, green, True)
    box(margin, h-47, w-2*margin, 0.55, (0.82, 0.88, 0.90))
    text(tag, margin, h-36, w-2*margin, 8.5, muted)
    text(f"{page_count:02d}", w-margin-22, h-36, 22, 8.5, muted, True)

def table(rows, y):
    count = len(rows[0])
    assert all(len(row) == count for row in rows), rows
    fractions = {
        2: [0.23,0.77], 3:[0.49,0.20,0.31],
        4:[0.25,0.25,0.25,0.25], 5:[0.24,0.19,0.19,0.19,0.19]
    }[count]
    if rows[0][0] == "Stage":
        fractions = [0.16,0.30,0.27,0.27]
    elif rows[0][0] == "Verification check":
        fractions = [0.25,0.11,0.11,0.53]
    elif rows[0][0] == "Group member":
        fractions = [0.23,0.18,0.59]
    elif rows[0][0] == "Quality":
        fractions = [0.16,0.24,0.29,0.31]
    columns = [width * f for f in fractions]
    for j, row in enumerate(rows):
        color = (1,1,1) if j == 0 else navy
        row_height = max(text(v, 0, 0, cw - 14, 9.3, color, j==0, False)
                         for v, cw in zip(row, columns)) + 8
        box(margin, y, width, row_height, navy if j == 0 else pale if j % 2 else (1,1,1))
        x = margin
        for value, column in zip(row, columns):
            text(value, x+7, y+4, column-14, 9.3, color, j==0)
            x += column
        y += row_height
    box(margin, y, width, 0.6, (0.84,0.90,0.91))
    return y+14

def recall_figure(y):
    reports = Path("eval/results/minikube")
    def recall(run):
        capture = next(reports.glob(f"gemini-{run}-*/offline/retrieval_eval_report.json"))
        return json.loads(capture.read_text())["clean_baseline"]["recall_at_5"]

    checks = json.loads((reports / "final-retrieval-gate.json").read_text())["details"]
    floor = next(check["threshold"] for check in checks if check["rule"] == "min_recall_at_5")
    x = margin + 77
    span = width - 122
    y += text("Figure 1 · Offline Recall@5", margin, y, width, 10, green, True) + 6
    for index, run in enumerate(("baseline", "candidate")):
        label_y = y + index * 19
        value = recall(run)
        text(run.capitalize(), margin, label_y, 70, 9, navy)
        box(x, label_y+1, span, 11, pale)
        box(x, label_y+1, value*span, 11, muted if index == 0 else green)
        text(f"{value:.2%}", x+span+6, label_y, 39, 9, navy, True)
    box(x+floor*span, y-1, 1, 34, navy)
    text(f"{floor:.0%} gate minimum", x+floor*span-30, y+34, 105, 8, muted)
    return y+52

def render_body(section, i):
    page(pw, ph, "Assignment 03  /  Report")
    lines = section.strip().splitlines()
    y = 70
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line:
            continue
        if line == "<!-- chart: recall5 -->":
            y = recall_figure(y)
        elif line.startswith("|"):
            rows = []
            while line.startswith("|"):
                row = [value.strip() for value in line.strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", v) for v in row):
                    rows.append(row)
                if idx >= len(lines) or not lines[idx].lstrip().startswith("|"):
                    break
                line = lines[idx].strip()
                idx += 1
            y = table(rows, y)
        elif line.startswith("# "):
            y += text(line[2:], margin, y, width, 32 if i==0 else 23, navy, True) + 15
        elif line.startswith("## "):
            y += text(line[3:], margin, y, width, 17, green, True) + 13
        elif line.startswith("### "):
            y += 4
            y += text(line[4:], margin, y, width, 12, green, True) + 6
        else:
            note = line.startswith(("Evidence ·", "Repository ·", "AI use ·"))
            y += text(line, margin, y, width, 9.5 if note else 10.5,
                      muted if note else navy) + (3 if re.match(r"^\d+\. ", line) else 9)
    if y >= ph-56:
        raise RuntimeError(f"Body page {i+1} overflows at {y}")
    print(f"Body page {i+1}: ends at {y:.0f}pt")
    c.show_page()

parts = re.split(r"\n---\n", source.read_text())
assert len(parts) <= 8
for i, part in enumerate(parts):
    render_body(part, i)

captions = {
    "A1": ("Batch B verification report", "Live report on the Batch B PVC. Each check shows pass or fail and its reason."),
    "A2": ("Repeat ingestion", "Completed Job logs show 0 embeddings computed and 16 retained chunks."),
    "B1": ("Evaluation at all three levels", "Saved Minikube report summaries. Gemini fallbacks and missing reviews limit the answer evaluation."),
    "C1": ("Candidate rejected", "The full release precheck blocked promotion. The offline retrieval checks also rejected the candidate."),
    "D1": ("Complete query trace", "The saved Batch B query trace, printed as one JSON record."),
    "D2": ("Prompt reconstruction by trace ID", "The tool prints the saved context and exact model request, including the system and user prompts."),
    "D3": ("Trace metrics and guardrail action", "Trace window statistics and the Batch B passage discarded before generation."),
    "E1": ("Coverage before and after", "The team's committed run at origin/dev compares 28 and 34 golden questions."),
    "F1": ("Namespace inventory", "Live Minikube object inventory for namespace rahbar."),
    "F2a": ("Completed ingestion Job", "Live logs from the completed Batch B ingestion Job."),
    "F2b": ("Completed evaluation Job", "Final baseline evaluation Job logs. Answer quality review and release remain pending."),
    "F2c": ("ConfigMap threshold demonstration", "Identical image and input score. The edited threshold flipped the diagnostic outcome."),
    "F3a": ("Rollout status", "Current live and candidate readiness, plus retained resource rollout output. Candidate promotion remains pending."),
    "F3b": ("Rollback and restored trace", "Terminal re-display of the saved rollout and undo output, with fields from the actual post-rollback trace."),
    "F4": ("Metrics and storage persistence", "Captured metrics endpoint output and Chroma PVC evidence from the Minikube run."),
}
page(pw, ph, "Assignment 03  /  Appendix index")
y=73
y += text("Appendix · Terminal evidence", margin,y,width,25,navy,True)+16
y += text("Terminal screenshots show commands and their output. Long outputs continue on the next labeled page. Archived evidence is identified in its caption.",margin,y,width)+15
for key, (title, _) in captions.items():
    y += text(f"{key}  ·  {title}",margin,y,width,10.5)+5
y += 13
y += text("Pending captures",margin,y,width,13,green,True)+8
text("C2  ·  A candidate passes the full release gate.\nF3c  ·  The passing candidate is promoted and the live rollout completes.",margin,y,width)
c.show_page()

for prefix, (title, description) in captions.items():
    paths = sorted(shots.glob(f"{prefix}.png")) + sorted(shots.glob(f"{prefix}-*.png"))
    assert paths, prefix
    for idx, image_path in enumerate(paths, 1):
        w, h = ph, pw
        page(w, h, f"Assignment 03  /  Appendix  /  {prefix}")
        suffix = f" ({idx}/{len(paths)})" if len(paths)>1 else ""
        text(f"{prefix}  ·  {title}{suffix}", margin, 57, w-2*margin, 19, navy, True)
        text(description, margin, 83, w-2*margin, 10, muted)
        img = cairo.ImageSurface.create_from_png(str(image_path))
        scale = min((w-2*margin)/img.get_width(), (h-153)/img.get_height())
        c.save()
        c.translate(margin, 104)
        c.scale(scale, scale)
        c.set_source_surface(img, 0, 0)
        c.paint()
        c.restore()
        c.show_page()

s.finish()
print(f"Exported {out} with {page_count} pages ({len(parts)} report pages)")
