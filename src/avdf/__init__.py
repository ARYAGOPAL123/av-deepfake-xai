"""Explainable and Uncertainty-Aware Audio-Visual Deepfake Detection (avdf)."""
__version__ = "1.0.0"
LABELS = ["real", "fake"]
REAL, FAKE = 0, 1

try:  # verify HTTPS (Hugging Face / timm weight downloads) against the OS certificate store, e.g. behind a corporate proxy
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
