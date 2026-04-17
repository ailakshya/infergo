"""infergo native Python SDK — zero-overhead inference via direct C bindings."""
from .core import LLM, Embedding, VectorDB, BM25, LoRA

try:
    from .uds_client import UDSClient
except ImportError:
    pass

try:
    from .shm_client import SHMClient
except ImportError:
    pass

__all__ = ["LLM", "Embedding", "VectorDB", "BM25", "LoRA", "UDSClient", "SHMClient"]
__version__ = "1.0.0"
