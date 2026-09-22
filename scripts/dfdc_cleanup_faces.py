"""Delete face crops only after the complete frozen embedding cache exists."""
import argparse
from pathlib import Path

import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--embedding-dir", required=True)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest)
    cache = Path(args.cache)
    embedding_dir = Path(args.embedding_dir)
    missing = []
    for clip_id in manifest.clip_id:
        path = embedding_dir / f"{clip_id}.pt"
        if not path.exists():
            missing.append(str(clip_id))
            continue
        item = torch.load(path, map_location="cpu", weights_only=True)
        if "visual" not in item or "audio" not in item:
            missing.append(str(clip_id))
    if missing:
        raise SystemExit(f"Refusing to delete face crops: {len(missing)} embeddings are missing visual/audio features")
    faces = list(cache.glob("*_faces.pt"))
    if not args.yes:
        print(f"Verified {len(manifest)} complete embeddings; {len(faces)} face crops would be deleted. Re-run with --yes.")
        return
    for path in faces:
        path.unlink()
    print(f"Verified {len(manifest)} complete embeddings; deleted {len(faces)} face crops from {cache}")


if __name__ == "__main__":
    main()
