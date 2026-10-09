"""
==============================================================================
⚙️ EXPERIMENT CONFIGURATION & CONTROL PANEL
==============================================================================
Cost-Aware Automated Failure Attribution in LLM Multi-Agent Systems
(FOSE7901 Placement Plan 2)

All settings across the entire project can be adjusted right here.
"""

import os
from pathlib import Path
from src.environment import load_environment
from src.runtime import isolate_gpu_zero

load_environment()  # Credentials/settings loaded before Hub, model or CUDA imports.

# ============================================================================
# 1. HARDWARE & GPU SETTINGS
# ============================================================================
# Physical GPU 0 is fixed for this shared server. Resolve its UUID before CUDA
# starts so CUDA enumeration cannot accidentally select a different device.
TARGET_GPU = "0"
PHYSICAL_GPU_UUID = isolate_gpu_zero()

# ============================================================================
# 2. EXPERIMENT PROFILE (One-Switch Reconfiguration)
# ============================================================================
# Choose an execution profile:
#
#   "quick_test"   : 1 small model (llama-8b), 1 run, 5% of data (~9 traces).
#                    Inference-only environment check; duration depends on input length.
#
#   "test_run"     : 2 models (llama-8b -> llama-70b), 3 runs, 10% data (~18 traces).
#                    Tests the complete cascade workflow on a small subset.
#
#   "openai_test"  : llama-8b -> gpt-5.6-sol, 3 runs, 10% data (~18 traces).
#   "full_run"     : All 5 models, 3 runs, 100% data (all 184 traces).
#                    The complete benchmark for your research report.
#
#   "custom"       : Uses the detailed manual settings in Section 3 below.
#
PROFILE = os.environ.get("EXPERIMENT_PROFILE", "full_run")

# ============================================================================
# 3. MANUAL / CUSTOM SETTINGS (Active when PROFILE = "custom")
# ============================================================================
CUSTOM_MODELS = ["llama-8b", "qwen-7b", "llama-70b", "qwen-72b", "gpt-5.6-sol"]
CUSTOM_SAMPLE_FRACTION = 1.0     # 1.0 = 100% (184 traces), 0.1 = 10% (~18 traces), 0.5 = 50%
CUSTOM_NUM_RUNS = 3              # Runs per model per trace for majority vote
CUSTOM_TEMPERATURE = 0.7         # Sampling temperature (> 0 for run variance)
CUSTOM_AUTO_DELETE_WEIGHTS = True  # Automatically purge model weights from HDD after finishing

# ============================================================================
# 4. PROFILE RESOLVER LOGIC
# ============================================================================
if PROFILE == "quick_test":
    ACTIVE_MODELS = ["llama-8b"]
    SAMPLE_FRACTION = 0.05       # ~9 traces
    NUM_RUNS = 1                 # 1 run
    TEMPERATURE = 0.7
    AUTO_DELETE_WEIGHTS = True
elif PROFILE == "test_run":
    ACTIVE_MODELS = ["llama-8b", "llama-70b"]
    SAMPLE_FRACTION = 0.10       # ~18 traces
    NUM_RUNS = 3                 # 3 runs for majority voting
    TEMPERATURE = 0.7
    AUTO_DELETE_WEIGHTS = True
elif PROFILE == "openai_test":
    ACTIVE_MODELS = ["llama-8b", "gpt-5.6-sol"]
    SAMPLE_FRACTION = 0.10
    NUM_RUNS = 3
    TEMPERATURE = 0.7  # Local models only; API uses its default sampling policy.
    AUTO_DELETE_WEIGHTS = True
elif PROFILE == "full_run":
    ACTIVE_MODELS = ["llama-8b", "qwen-7b", "llama-70b", "qwen-72b", "gpt-5.6-sol"]
    SAMPLE_FRACTION = 1.0        # All 184 traces
    NUM_RUNS = 3                 # 3 runs for majority voting
    TEMPERATURE = 0.7
    AUTO_DELETE_WEIGHTS = True
elif PROFILE == "custom":
    ACTIVE_MODELS = CUSTOM_MODELS
    SAMPLE_FRACTION = CUSTOM_SAMPLE_FRACTION
    NUM_RUNS = CUSTOM_NUM_RUNS
    TEMPERATURE = CUSTOM_TEMPERATURE
    AUTO_DELETE_WEIGHTS = CUSTOM_AUTO_DELETE_WEIGHTS
else:
    raise ValueError(f"Unknown PROFILE: {PROFILE}.")

# ============================================================================
# 5. DATASET & VALIDATION SPLIT
# ============================================================================
VAL_RATIO = 0.6                  # 60% validation (for threshold tuning), 40% test
RANDOM_SEED = 42                 # Deterministic seed for reproducible splits

# ============================================================================
# 6. THRESHOLD SWEEP PARAMETERS
# ============================================================================
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.01
THRESHOLD_STEP = 0.05

# ============================================================================
# 7. OUTPUT FORMATTING & EXPORT SETTINGS
# ============================================================================
SAVE_PNG = True                  # Export high-res PNG figures (300 DPI)
SAVE_PDF = True                  # Export publication-quality vector PDFs for LaTeX
EXPORT_MARKDOWN = True           # Export copy-paste ready Markdown tables (.md)
EXPORT_LATEX = True              # Export copy-paste ready LaTeX tables (.tex)

# ============================================================================
# 8. DIRECTORY PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("EXPERIMENT_RESULTS_DIR", str(PROJECT_ROOT / "results" / PROFILE))).resolve()
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
AGGREGATED_RESULTS_DIR = RESULTS_DIR / "aggregated"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"

# Subfolders in results/analysis/ for clean organization
TABLES_DIR = ANALYSIS_DIR / "tables"       # CSV, Markdown (.md), and LaTeX (.tex)
FIGURES_DIR = ANALYSIS_DIR / "figures"     # 300 DPI PNG & vector PDF plots
REPORTS_DIR = ANALYSIS_DIR / "reports"     # Executive summary reports

# ============================================================================
# 9. MODEL REGISTRY
# ============================================================================
MODELS = {
    "llama-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "quantisation": None,           # BF16; context adds KV-cache/prefill memory
        "max_new_tokens": 512,
        "theoretical_cost_per_1m_input": 0.10,    # USD
        "theoretical_cost_per_1m_output": 0.30,
    },
    "qwen-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "quantisation": None,           # BF16; context adds KV-cache/prefill memory
        "max_new_tokens": 512,
        "theoretical_cost_per_1m_input": 0.10,
        "theoretical_cost_per_1m_output": 0.30,
    },
    "llama-70b": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "quantisation": "4bit",         # NF4 weights; BF16 compute; context adds memory
        "max_new_tokens": 512,
        "theoretical_cost_per_1m_input": 0.90,
        "theoretical_cost_per_1m_output": 0.90,
    },
    "qwen-72b": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "quantisation": "4bit",         # NF4 weights; BF16 compute; context adds memory
        "max_new_tokens": 512,
        "theoretical_cost_per_1m_input": 0.90,
        "theoretical_cost_per_1m_output": 0.90,
    },
    "gpt-5.6-sol": {
        "provider": "openai",
        "api_model": "gpt-5.6-sol",
        "reasoning_effort": os.environ.get("OPENAI_REASONING_EFFORT", "medium"),
        "max_new_tokens": 16384,  # Includes hidden reasoning and visible output.
        "context_window": 1050000,
        "service_tier": "default",
        "sampling_policy": "api_default_no_seed",
        "theoretical_cost_per_1m_input": 4.0,
        "theoretical_cost_per_1m_cached_input": 0.40,
        "cache_write_multiplier": 1.25,
        "theoretical_cost_per_1m_output": 20.0,
        "long_context_threshold": 272000,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
        "pricing_date": "2026-10-10",
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "cost_basis": "API usage priced at frozen published rates; not an invoice",
    },
}
if MODELS['gpt-5.6-sol']['reasoning_effort'] not in ('none', 'low', 'medium', 'high', 'xhigh', 'max'):
    raise ValueError('Invalid OPENAI_REASONING_EFFORT.')

# ============================================================================
# 10. CASCADE PAIRS
# ============================================================================
ALL_CASCADE_PAIRS = [
    ("llama-8b", "llama-70b"),
    ("llama-8b", "qwen-72b"),
    ("qwen-7b", "llama-70b"),
    ("qwen-7b", "qwen-72b"),
    ("llama-8b", "gpt-5.6-sol"),
    ("qwen-7b", "gpt-5.6-sol"),
]
CASCADE_PAIRS = [(cheap, strong) for cheap, strong in ALL_CASCADE_PAIRS
                 if cheap in ACTIVE_MODELS and strong in ACTIVE_MODELS]
MIN_ACCURACY_RETENTION = 0.95
ALL_STRONG_THRESHOLD = 1.000001
TOP_P = 0.9
TOP_K = 50
for _alias, _model in MODELS.items():
    if _model.get('provider') == 'openai':
        continue
    _model['provider'] = 'huggingface'
    _model['cost_basis'] = 'illustrative API-equivalent local cost'
    _model["revision"] = "main"
    _model["context_window"] = 131072
    if _alias.startswith("qwen"):
        _model["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0,
                                 "original_max_position_embeddings": 32768}


def print_config_banner(stage_name: str = "") -> None:
    """Print a clean visual banner showing active configuration."""
    title = f" STAGE: {stage_name} " if stage_name else " EXPERIMENT CONFIGURATION "
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70)
    print(f" - Active Profile   : {PROFILE}")
    print(f" - Target GPU       : cuda:{TARGET_GPU} (isolated from other GPUs)")
    print(f" - Physical GPU UUID: {PHYSICAL_GPU_UUID or 'unresolved; server preflight required'}")
    print(f" - Results Directory: {RESULTS_DIR}")
    print(f" - Active Models    : {', '.join(ACTIVE_MODELS)}")
    print(f" - Dataset Fraction : {SAMPLE_FRACTION * 100:.1f}% (~{int(round(184 * SAMPLE_FRACTION))} traces)")
    print(f" - Runs per Trace   : {NUM_RUNS} (local temperature: {TEMPERATURE})")
    if 'gpt-5.6-sol' in ACTIVE_MODELS:
        print(f" - OpenAI Reasoning : {MODELS['gpt-5.6-sol']['reasoning_effort']} (API default sampling; no seed)")
    print(f" - Auto-Purge HDD   : {AUTO_DELETE_WEIGHTS} (model files removed upon completion)")
    print("=" * 70 + "\n")

