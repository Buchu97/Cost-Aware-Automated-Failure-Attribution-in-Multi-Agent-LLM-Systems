"""Load exactly one local model on the sole visible GPU and release it safely."""
import gc
import shutil
from pathlib import Path
from src.progress import heartbeat


def load_tokenizer(model_config: dict, cache_dir: str | Path | None = None):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config["hf_id"], cache_dir=str(cache_dir) if cache_dir else None,
        revision=model_config.get("revision", "main"), trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_config: dict, cache_dir: str | Path | None = None) -> tuple:
    import config
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
    from src.runtime import validate_gpu
    if not config.PHYSICAL_GPU_UUID:
        raise RuntimeError("Cannot resolve physical GPU 0 with nvidia-smi. Run preflight first.")
    validate_gpu(torch, config.PHYSICAL_GPU_UUID)
    tokenizer = load_tokenizer(model_config, cache_dir)
    kwargs = dict(device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
        cache_dir=str(cache_dir) if cache_dir else None, revision=model_config.get("revision", "main"))
    if model_config.get("quantisation") == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    if model_config.get("rope_scaling"):
        hf_config = AutoConfig.from_pretrained(model_config["hf_id"], cache_dir=kwargs["cache_dir"], revision=kwargs["revision"])
        hf_config.rope_scaling = model_config["rope_scaling"]
        hf_config.max_position_embeddings = model_config["context_window"]
        kwargs["config"] = hf_config
    with heartbeat(f"Download/load {model_config['hf_id']} onto cuda:0 (physical GPU 0)"):
        model = AutoModelForCausalLM.from_pretrained(model_config["hf_id"], **kwargs)
    model.eval()
    free, total = torch.cuda.mem_get_info(0)
    allocated = torch.cuda.memory_allocated(0)
    print(f'[GPU 0] model ready | allocated={allocated / 1024**3:.1f} GiB | '
          f'free={free / 1024**3:.1f}/{total / 1024**3:.1f} GiB', flush=True)
    return model, tokenizer


def release_cuda_memory() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()


class ModelSession:
    """Own all weight references; clear them before the next model is loaded."""
    def __init__(self, model_config: dict, cache_dir: Path):
        self.model_config, self.cache_dir = model_config, cache_dir
        self.model = self.tokenizer = None

    def __enter__(self):
        try:
            self.model, self.tokenizer = load_model(self.model_config, self.cache_dir)
        except BaseException:
            release_cuda_memory()
            raise
        return self

    def __exit__(self, *exception_info):
        self.model = self.tokenizer = None
        gc.collect()
        release_cuda_memory()


def delete_model_from_disk(cache_dir: str | Path, cache_root: str | Path | None = None) -> None:
    import config
    root = Path(cache_root or config.MODEL_CACHE_DIR).resolve()
    candidate = Path(cache_dir)
    target = candidate.resolve()
    if candidate.is_symlink() or target == root or not target.is_relative_to(root):
        raise ValueError(f"Refusing to delete outside a model-specific cache: {target}")
    if target.exists():
        shutil.rmtree(target)
        print(f"[model_loader] Purged model cache: {target}")
