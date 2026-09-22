"""Build a clearly labelled manifest from clips whose CPU cache is complete."""
from pathlib import Path
import pandas as pd
from avdf.ingest import group_stratified_split

source = Path('data/manifests/dfdc.csv')
cache = Path('data/cache/dfdc_16f')
out = Path('data/manifests/dfdc_fast.csv')
df = pd.read_csv(source)
complete = df[df.clip_id.map(lambda c: (cache / f'{c}_faces.pt').exists() and (cache / f'{c}.wav').exists())].copy()
if complete.empty or complete.label.nunique() < 2:
    raise SystemExit(f'Need both labels in complete cache; found {len(complete)} rows')
complete = group_stratified_split(complete, seed=42)
out.parent.mkdir(parents=True, exist_ok=True)
complete.to_csv(out, index=False)
print(f'complete clips: {len(complete)}; source groups: {complete.subject.nunique()}')
print(complete.groupby(['split', 'label']).size().to_string())
print(f'group overlap: {complete.groupby("subject").split.nunique().gt(1).sum()}')
print(f'saved: {out}')
