"""Forensic case report (A4 PDF) rendered with matplotlib, so no extra dependency is needed."""
import io
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

INK, MUTED, LINE = "#16212b", "#5b6b7a", "#d5dde5"
ACCENT, FAKE, REAL, WARN = "#1f5f99", "#c2410c", "#15803d", "#a16207"
A4 = (8.27, 11.69)
DASH = "—"


def _fmt(v, digits=3):
    if v is None or v == "":
        return DASH
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _header(fig, title, case_id, page, pages):
    fig.patches.append(mpatches.Rectangle((0, 0.935), 1, 0.065, transform=fig.transFigure, color=INK, zorder=0))
    fig.text(0.06, 0.965, "AVDF · Audio-Visual Deepfake Forensics", color="white", fontsize=11, weight="bold", va="center")
    fig.text(0.94, 0.965, title, color="#c9d6e2", fontsize=9, ha="right", va="center")
    fig.text(0.06, 0.03, f"Case {case_id}", color=MUTED, fontsize=7.5)
    fig.text(0.94, 0.03, f"Page {page} of {pages}", color=MUTED, fontsize=7.5, ha="right")
    fig.lines.append(mlines.Line2D([0.06, 0.94], [0.045, 0.045], transform=fig.transFigure, color=LINE, lw=0.8))


def _rows(fig, y, rows, label_w=0.25):
    for label, value in rows:
        fig.text(0.06, y, label, color=MUTED, fontsize=8.5, va="top")
        lines = textwrap.wrap(_fmt(value), 70) or [DASH]
        fig.text(0.06 + label_w, y, "\n".join(lines), color=INK, fontsize=8.5, va="top",
                 family="monospace" if label in ("SHA-256", "File") else None)
        y -= 0.022 * len(lines) + 0.008
    return y


def _gauge(fig, x, y, w, label, value, threshold, colour, higher_is_better):
    ax = fig.add_axes([x, y, w, 0.018])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.barh(0.5, 1, height=1, color="#eef2f6")
    if value is not None:
        ax.barh(0.5, min(max(value, 0), 1), height=1, color=colour)
    ax.axvline(threshold, color=INK, lw=1.4)
    fig.text(x, y + 0.028, label, fontsize=8, color=MUTED)
    fig.text(x + w, y + 0.028, f"{_fmt(value)}   threshold {threshold}", fontsize=8, color=INK, ha="right")
    ok = value is not None and ((value >= threshold) if higher_is_better else (value <= threshold))
    fig.text(x, y - 0.018, "within threshold" if ok else "outside threshold", fontsize=7, color=REAL if ok else WARN)


def _image(fig, rect, path, title):
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)
    p = Path(path) if path else None
    if p and p.exists():
        ax.imshow(mpimg.imread(str(p)))
    else:
        ax.text(0.5, 0.5, "Not available for this case", ha="center", va="center", color=MUTED, fontsize=8.5,
                transform=ax.transAxes)
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=LINE, ls="--"))


def build_pdf(case, tau, u_max):
    """case: dict with verdict/confidence/uncertainty/... (image fields are filesystem paths). Returns PDF bytes."""
    case_id = str(case.get("result_id", DASH))
    verdict = case.get("verdict")
    conf, unc = case.get("confidence"), case.get("uncertainty")
    has_images = any(case.get(k) for k in ("faces_preview", "gradcam", "shap"))
    pages = 2 if has_images else 1
    buf = io.BytesIO()
    with PdfPages(buf, metadata={"Title": f"AVDF forensic report {case_id}", "Author": "AVDF"}) as pdf:
        fig = plt.figure(figsize=A4)
        _header(fig, "Forensic analysis report", case_id, 1, pages)
        fig.text(0.06, 0.895, "Case summary", fontsize=18, color=INK, weight="bold")
        fig.text(0.06, 0.872, f"Generated {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')}  ·  "
                              f"{case.get('mode', DASH)}", fontsize=8.5, color=MUTED)
        if case.get("demo_fixture"):
            fig.patches.append(mpatches.FancyBboxPatch((0.06, 0.822), 0.88, 0.036, transform=fig.transFigure,
                                                       boxstyle="round,pad=0.004", fc="#fef3c7", ec="#f59e0b"))
            fig.text(0.075, 0.840, f"DEMO FIXTURE {DASH} illustrates the workflow only. This is not a model prediction "
                                   "and must not be cited as a result.", fontsize=8.2, color="#78350f", va="center")
        # verdict card
        top = 0.79
        colour = FAKE if verdict == "fake" else REAL if verdict == "real" else MUTED
        fig.patches.append(mpatches.FancyBboxPatch((0.06, top - 0.115), 0.88, 0.115, transform=fig.transFigure,
                                                   boxstyle="round,pad=0.004", fc="#f7f9fb", ec=LINE))
        fig.text(0.085, top - 0.025, "VERDICT", fontsize=7.5, color=MUTED, weight="bold")
        fig.text(0.085, top - 0.075, (verdict or "unavailable").upper(), fontsize=30, color=colour, weight="bold")
        fig.text(0.55, top - 0.025, "ROUTING DECISION", fontsize=7.5, color=MUTED, weight="bold")
        decision = case.get("decision") or DASH
        fig.text(0.55, top - 0.057, decision, fontsize=14, color=WARN if decision == "HUMAN REVIEW" else INK, weight="bold")
        fig.text(0.55, top - 0.07, "\n".join(textwrap.wrap(case.get("reason") or "", 52)[:3]), fontsize=7.5,
                 color=MUTED, va="top")
        # gauges
        _gauge(fig, 0.06, 0.60, 0.40, "Confidence (max class probability)", conf, tau, ACCENT, True)
        _gauge(fig, 0.54, 0.60, 0.40, "Predictive uncertainty", unc, u_max, WARN, False)
        # details table
        fig.text(0.06, 0.54, "Case details", fontsize=12, color=INK, weight="bold")
        y = _rows(fig, 0.515, [
            ("File", case.get("display_name") or case.get("file") or case.get("file_path")),
            ("SHA-256", case.get("sha256") or case.get("input_hash")),
            ("Analysed at", case.get("created_at")),
            ("Model", case.get("model") or case.get("model_version")),
            ("Uncertainty method", case.get("method")),
            ("P(fake)", case.get("p_fake")),
            ("Frames analysed", case.get("frames_analysed")),
            ("Audio analysed (s)", case.get("audio_sec") or case.get("duration_sec")),
            ("Review status", case.get("review_status") or ("routed" if case.get("review") else "not routed")),
        ])
        history = case.get("review_history") or []
        if history:
            fig.text(0.06, y - 0.01, "Review trail", fontsize=12, color=INK, weight="bold")
            _rows(fig, y - 0.035, [(h.get("created_at", ""), f"{h.get('status', '').upper()} by "
                                   f"{h.get('reviewer') or DASH}: {h.get('note') or ''}") for h in history])
        fig.text(0.06, 0.085, "\n".join(textwrap.wrap(
            "Method: bidirectional cross-modal attention fusion of face (visual) and speech (audio) embeddings. "
            f"A case is auto-accepted only when confidence >= tau ({tau}) and uncertainty <= u_max ({u_max}); "
            "otherwise it is routed to a human reviewer. Automated verdicts are decision support, not proof.", 120)),
            fontsize=7, color=MUTED, va="top")
        pdf.savefig(fig); plt.close(fig)

        if has_images:
            fig = plt.figure(figsize=A4)
            _header(fig, "Evidence", case_id, 2, pages)
            fig.text(0.06, 0.895, "Explainability evidence", fontsize=18, color=INK, weight="bold")
            _image(fig, [0.06, 0.66, 0.88, 0.19], case.get("faces_preview"), "Extracted face crops (input to the visual branch)")
            _image(fig, [0.06, 0.40, 0.88, 0.21], case.get("gradcam"),
                   f"Visual evidence {DASH} Grad-CAM through the fused model (warm = pushes towards FAKE)")
            _image(fig, [0.06, 0.10, 0.88, 0.25], case.get("shap"),
                   f"Audio evidence {DASH} KernelSHAP over Mel band × time segment (+ pushes towards FAKE)")
            pdf.savefig(fig); plt.close(fig)
    return buf.getvalue()
