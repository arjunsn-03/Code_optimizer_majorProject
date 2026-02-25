"""
Central configuration for the Carbon-Aware Code Optimizer.
All tuneable parameters live here; override via environment variables.
"""
import os
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Ollama (local LLM) ────────────────────────────────────────────────────────
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL    = os.getenv("DEFAULT_MODEL", "qwen3")
AVAILABLE_MODELS = ["qwen3", "llama3", "mistral"]

# ── Embeddings (local, sentence-transformers) ─────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── Vector / Artifact store ───────────────────────────────────────────────────
VECTOR_DB_PATH = ROOT / "dataset" / "faiss_index"
METADATA_PATH  = ROOT / "dataset" / "metadata.json"
RESULTS_PATH   = ROOT / "dataset" / "results.json"

# ── Energy & CO₂ ─────────────────────────────────────────────────────────────
# World-average grid intensity (kg CO₂e per kWh).  Override for your region.
GRID_INTENSITY_KG_PER_KWH = float(os.getenv("GRID_INTENSITY", "0.475"))
# Assumed CPU TDP in watts (laptop default).  Override for desktop / server.
CPU_TDP_WATTS = float(os.getenv("CPU_TDP_WATTS", "15"))

# ── Profiler ──────────────────────────────────────────────────────────────────
WARMUP_RUNS = int(os.getenv("WARMUP_RUNS", "2"))
BENCH_RUNS  = int(os.getenv("BENCH_RUNS", "5"))

# ── Supported languages ───────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".go":   "go",
    ".java": "java",
    ".cpp":  "cpp",
    ".c":    "c",
    ".rs":   "rust",
}

# ── Verifier & pipeline ───────────────────────────────────────────────────────
TEST_TIMEOUT_SECS = int(os.getenv("TEST_TIMEOUT_SECS", "120"))
MAX_PATCH_RISK    = float(os.getenv("MAX_PATCH_RISK", "0.7"))
TOP_N_HOTSPOTS    = int(os.getenv("TOP_N_HOTSPOTS", "5"))
