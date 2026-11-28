import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("Vte", "2.91")
from gi.repository import Gdk, GLib, Gtk, Pango, Vte

out = Path("docs/evidence/assignment3-report/screenshots")
k = "kubectl --context minikube -n rahbar"
ev = "docs/evidence/assignment3-minikube"
base = "eval/results/minikube/gemini-baseline-eval-gemini-baseline-cqwlr"
a1 = "minikube ssh 'sudo cat /tmp/hostpath-provisioner/rahbar/batch-b-corpus/quarantine/verification_report.json' < /dev/null"
trace = "f4428b49e01c4219b6805e2f87755861"
commands = {
    "A1": a1 + """ | jq -r '([.run_id, .total_documents, .passed_count, .quarantined_count] | @tsv), (.results[] | .source, (.checks[] | ["  ", .name, (if .passed then "PASS" else "FAIL" end), .reason] | @tsv))'""",
    "A2": f"{k} logs job/ingest-batch-b-repeat",
    "B1": rf"""jq -r '.summary | "Embedding | model=\(.model) | paraphrase mean=\(.paraphrase_mean_sim) | unrelated mean=\(.unrelated_mean_sim) | gap=\(.separation_gap) | time/chunk=\(.latency_mean_ms) ms"' {base}/offline/embedding_eval_report.json
jq -r '.clean_baseline | "Retrieval | recall@3=\(.recall_at_3) | recall@5=\(.recall_at_5) | MRR=\(.mrr) | nDCG@5=\(.ndcg_at_5)"' {base}/offline/retrieval_eval_report.json
jq -r '"Answers | questions=\(.questions) | citation validity=\(.citation_validity) | abstention=\(.abstention_correctness) | p50=\(.latency_p50_seconds)s | p95=\(.latency_p95_seconds)s"' {base}/answers-1.json
jq -r '"Guardrail for release | unreviewed answers=\(length)"' {base}/review-needed-1.json
{k} logs job/eval-gate-candidate-precheck""",
    "C1": f"""{k} logs job/eval-gate-candidate-precheck
cat {ev}/final-retrieval-gate.log
cat {ev}/candidate-release-rejection.log""",
    "D1": "jq -c . traces/batch-b-injection.json",
    "D2": f"""trace_view=$(mktemp -d)
jq -c . traces/batch-b-injection.json > "$trace_view/trace.jsonl"
python3 scripts/query_traces.py "$trace_view" --id {trace} | jq -c '{{trace_id, context, model_inputs}}'""",
    "D3": f"""cat {ev}/summary.json
jq '{{trace_id, blocked_chunk_ids, guardrails}}' traces/batch-b-injection.json""",
    "E1": """git show origin/dev:eval/results/drift_experiment_report.json | jq '{model: .configuration.embed_model, similarity_threshold: .configuration.similarity_threshold, before: .metrics.baseline, after: .metrics.candidate, delta: .metrics.deltas, gate: .gate_decision}'""",
    "F1": f"{k} get all,pvc,configmap,cronjob,job",
    "F2a": f"{k} logs job/ingest-batch-b-demo",
    "F2b": f"{k} logs --tail=26 job/eval-gemini-baseline",
    "F2c": f"""{k} get jobs eval-gate-demo-fail eval-gate-demo-pass
{k} logs job/eval-gate-demo-fail
{k} logs job/eval-gate-demo-pass""",
    "F3a": f"""{k} rollout status deployment/api
{k} rollout status deployment/api-candidate
cat {ev}/measured-resource-rollouts.txt""",
    "F3b": f"""cat {ev}/bad-rollout.log {ev}/rollout-undo.txt {ev}/restored-rollout.txt
jq '{{trace_id, timestamp, collection, embedding_model, embedding_version}}' traces/post-rollback.json""",
    "F4": f"""sed -n '/^rahbar_/p' {ev}/metrics.txt
cat {ev}/chroma-persistence.txt""",
}

window = Gtk.OffscreenWindow()
terminal = Vte.Terminal()
terminal.set_size_request(1560, 864)
terminal.set_font(Pango.FontDescription("Monospace 10"))
terminal.set_colors(Gdk.RGBA(0.92,0.96,1,1), Gdk.RGBA(0.07,0.10,0.15,1), None)
terminal.set_scrollback_lines(10000)
window.add(terminal)
window.show_all()
window.resize(1560,864)

def update():
    while Gtk.events_pending():
        Gtk.main_iteration_do(False)

def pause():
    loop=GLib.MainLoop()
    def quit_loop():
        loop.quit()
        return False
    GLib.timeout_add(120, quit_loop)
    loop.run()

pause()
for label, command in commands.items():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as shell_script:
        shell_script.write(command + "\n")
        shell_script.flush()
        shell = "bash --noprofile --norc -v " + shlex.quote(shell_script.name)
        result = subprocess.run(["script", "-qec", shell, "/dev/null"], capture_output=True, timeout=30)
        if result.returncode:
            raise RuntimeError(f"{label}: {result.returncode}: {result.stdout[-1000:]!r}")
    terminal.reset(True, True)
    terminal.feed(result.stdout)
    pause()
    scroll = terminal.get_vadjustment()
    bottom = int(scroll.get_upper() - scroll.get_page_size())
    top = int(scroll.get_lower())
    step = max(1, int(scroll.get_page_size()) - 2)
    pages = list(range(top, bottom, step)) + [bottom] if bottom > top else [top]
    for index, position in enumerate(pages, 1):
        scroll.set_value(position)
        pause()
        update()
        destination = out / (f"{label}-{index:02}.png" if len(pages) > 1 else f"{label}.png")
        pixbuf = window.get_pixbuf()
        assert pixbuf.get_width() >= 1560 and pixbuf.get_height() >= 864
        pixbuf.savev(str(destination), "png", [], [])
    print(label, len(pages), terminal.get_column_count(), terminal.get_row_count(), flush=True)
