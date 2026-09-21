"""Analyst dashboard.   streamlit run dashboard/app.py -- --config configs/default.yaml"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avdf.pipeline import Pipeline  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.yaml")
args, _ = ap.parse_known_args()

st.set_page_config(page_title="Deepfake Forensics Dashboard", layout="wide")


@st.cache_resource
def get_pipe(cfg):
    return Pipeline(cfg)


pipe = get_pipe(args.config)
st.title("Deepfake Forensics Dashboard")
st.caption("Explainable & uncertainty-aware audio-visual deepfake detection")

tab1, tab2, tab3 = st.tabs(["Analyse clip", "Review queue", "Audit log"])

with tab1:
    up = st.file_uploader("Upload a video clip", type=["mp4", "avi", "mov", "mkv"])
    explain = st.checkbox("Generate explanations (Grad-CAM + SHAP)", value=True)
    if up is not None:
        dst = Path("results/uploads"); dst.mkdir(parents=True, exist_ok=True)
        path = dst / up.name; path.write_bytes(up.getbuffer())
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.video(str(path))
        with st.spinner("Analysing..."):
            pipe.explain = explain
            try:
                r = pipe.analyse(path)
            except ValueError as e:
                st.error(f"Invalid input: {e}"); st.stop()
        with c2:
            colour = "red" if r["verdict"] == "fake" else "green"
            st.markdown(f"## Verdict: :{colour}[{r['verdict'].upper()}]")
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{r['confidence']*100:.1f}%")
            m2.metric("Uncertainty", f"{r['uncertainty']:.3f}")
            m3.metric("Decision", r["decision"])
            st.progress(min(max(r["confidence"], 0.0), 1.0), text=f"Confidence vs threshold τ = {pipe.cfg.uncertainty.tau}")
            if r["review"]:
                st.warning("Low confidence / high uncertainty — case sent to the human review queue.")
        if explain:
            e1, e2 = st.columns(2)
            if r.get("gradcam"):
                e1.subheader("Visual evidence (Grad-CAM)"); e1.image(r["gradcam"])
            if r.get("shap"):
                e2.subheader("Audio evidence (SHAP)"); e2.image(r["shap"])
        with st.expander("Raw result"):
            st.json(r)

with tab2:
    rows = pipe.db.pending()
    st.subheader(f"{len(rows)} case(s) pending review")
    for row in rows:
        with st.container(border=True):
            st.write(f"**#{row['review_id']}** — {Path(row['file_path']).name} · verdict **{row['verdict']}** · "
                     f"confidence {row['confidence_score']:.2f} · uncertainty {row['uncertainty_score']:.3f}")
            b1, b2, b3 = st.columns(3)
            if b1.button("Approve", key=f"a{row['review_id']}"):
                pipe.db.resolve(row["review_id"], "approved"); st.rerun()
            if b2.button("Reject", key=f"r{row['review_id']}"):
                pipe.db.resolve(row["review_id"], "rejected"); st.rerun()
            if b3.button("Escalate to forensic team", key=f"e{row['review_id']}"):
                pipe.db.resolve(row["review_id"], "escalated"); st.rerun()

with tab3:
    df = pd.DataFrame(pipe.db.export())
    st.dataframe(df, use_container_width=True)
    if not df.empty:
        st.download_button("Export audit report (CSV)", df.to_csv(index=False), "audit_report.csv")
