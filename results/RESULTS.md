# DFDC Sample Results

Preliminary results on the DFDC sample (≈400 videos, REAL/FAKE labels only, mostly visual manipulations). Full FakeAVCeleb evaluation pending.

## Dataset verification

- Training sample: 400 `.mp4` files and 400 metadata entries.
- Labels: 77 REAL and 323 FAKE.
- Source groups: 244, using each FAKE video's `original` identifier for grouping.
- Leak-free split: train 54 REAL / 226 FAKE; validation 12 REAL / 49 FAKE; test 11 REAL / 48 FAKE.
- 265 FAKE videos reference originals absent from this downloaded sample. They remain grouped by source identifier; raw data was not modified.
- The unlabeled `test_videos` directory contains 400 videos and is reserved for demonstration only.

## Hardware and run status

- CPU: 11th Gen Intel Core i3-1115G4, 4 logical processors.
- RAM: approximately 7.7 GiB.
- CUDA: unavailable; Torch CPU runtime selected.
- Free disk observed before processing: approximately 210 GiB.
- A shared 16-frame CPU preprocessing cache was started. Model metrics, plots, explanations, and the live dashboard test remain pending until it completes.

No performance numbers are reported here until they are measured by the training and evaluation commands.