# Carbon-Aware Code Optimizer

> **B.Tech Final Year Project** — LLM-driven static + dynamic code analysis pipeline that identifies high-energy hotspots in a codebase and proposes correctness-preserving refactors to reduce energy consumption and CO₂ emissions.

---

## Features

| Feature | Detail |
|---|---|
| Multi-file project support | Walks an entire project, builds per-function hotspot scores |
| ML / AI project support | Detects framework imports, selects energy-aware ML prompts |
| Local LLM inference | Calls **qwen3**, **llama3**, or **mistral** via Ollama (no cloud required) |
| RAG context | FAISS vector store with code-aware embeddings for retrieval-augmented generation |
| Sandbox verification | Applies patch in an isolated temp directory, runs pytest before accepting |
| Energy measurement | CPU-time × TDP proxy on macOS; RAPL-compatible on Linux |
| CO₂ estimation | kWh × configurable grid intensity factor |
| Rich CLI | Three commands: `optimize`, `analyze`, `check` |

---

## Project structure

```
majomajoprojo/
├── cli.py                         # Entry point
├── config.py                      # All tunable settings
├── requirements.txt
├── .env.example
│
├── ingestor/
│   ├── clone.py                   # Repo walker / git clone
│   └── ast_extract.py             # Python AST → function metadata
│
├── static_analyzer/
│   └── analyzer.py                # Complexity + anti-pattern detection
│
├── profiler/
│   └── energy.py                  # CPU-time & energy measurement
│
├── indexer/
│   ├── embedder.py                # sentence-transformers embeddings
│   └── store.py                   # FAISS vector store
│
├── llm/
│   ├── client.py                  # Ollama HTTP wrapper
│   ├── prompts.py                 # Prompt templates (general + ML)
│   └── optimizer.py               # RAG + patch generation
│
├── verifier/
│   └── runner.py                  # Sandbox patch apply + pytest run
│
├── estimator/
│   └── energy.py                  # kWh → CO₂e + savings projection
│
├── pipeline/
│   └── orchestrator.py            # Full 5-stage pipeline
│
├── dataset/                       # Runtime artefacts (FAISS, results JSON)
├── tests/
│   ├── test_static_analyzer.py
│   └── test_estimator.py
└── ARCHITECTURE.md
```

---

## Quick start

### 1. Prerequisites

```bash
# Ollama must be installed and running
ollama serve &

# Pull the models you want to use (at least one)
ollama pull qwen3
ollama pull llama3
ollama pull mistral
```

### 2. Set up environment

```bash
conda activate majorproject

# Install all Python dependencies
pip install -r requirements.txt
```

### 3. Copy and edit environment variables (optional)

```bash
cp .env.example .env
# Edit .env to set CPU_TDP_WATTS for your machine, grid intensity, etc.
```

### 4. Verify Ollama connectivity

```bash
python cli.py check
```

### 5. Run static analysis on a project

```bash
# Quick hotspot scan — no LLM needed
python cli.py analyze /path/to/your/project
```

### 6. Run the full optimiser

```bash
# Full pipeline: ingest → analyse → optimise → verify → report
python cli.py optimize /path/to/your/project --model qwen3 --top-n 5

# Use llama3 and flag GPU presence for ML repos:
python cli.py optimize /path/to/ml/project --model llama3 --gpu
```

Results are saved to `dataset/results.json`.

---

## Running tests

```bash
conda activate majorproject
pytest tests/ -v
```

Tests cover the static analyser, AST extractor, and energy estimator and require **no LLM or network**.

---

## Configuration

All settings can be overridden by environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_MODEL` | `qwen3` | Default model |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `GRID_INTENSITY` | `0.475` | kg CO₂e / kWh (world avg) |
| `CPU_TDP_WATTS` | `15` | Laptop TDP; set to 65 for desktop |
| `MAX_PATCH_RISK` | `0.7` | Reject patches with risk score above this |
| `TOP_N_HOTSPOTS` | `5` | Hotspot functions to optimise per run |
| `BENCH_RUNS` | `5` | Benchmark repetitions |

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full DFD (L0–L2), flowchart, component specs, measurement protocol, and 12-week milestone plan.

---

## Energy measurement notes

- **macOS / Windows**: energy is estimated as `wall_time × (cpu_utilisation × TDP)`.  Set `CPU_TDP_WATTS` accurately for your CPU for best results.
- **Linux with RAPL**: swap `profiler/energy.py` to use `pyRAPL` for hardware-level measurements (install with `pip install pyRAPL`).
- **GPU**: pass `--gpu` flag; the LLM will suggest ML-specific optimisations.  For true GPU power readings integrate `pynvml` into `profiler/energy.py`.

---

## License

MIT — for academic use.
