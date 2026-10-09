"""Read project .env settings without executing or expanding their contents."""
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
CREDENTIAL_KEYS = ('HF_TOKEN', 'OPENAI_API_KEY')
ALLOWED_KEYS = set(CREDENTIAL_KEYS) | {
    'OPENAI_BUDGET_USD', 'OPENAI_REASONING_EFFORT', 'EXPERIMENT_PROFILE',
    'EXPERIMENT_RESULTS_DIR',
}


def dotenv_values(path: Path | None = None) -> dict[str, str]:
    """Support KEY=value, optional quotes, comments and UTF-8 BOM; no interpolation."""
    path = Path(path) if path is not None else ROOT / '.env'
    if not path.exists():
        return {}
    values = {}
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not separator or key not in ALLOWED_KEYS:
            raise ValueError(f'Invalid or unsupported .env setting at line {number}.')
        if value.startswith(('"', "'")):
            quote = value[0]
            end = value.find(quote, 1)
            if end == -1 or (value[end + 1:].strip() and not value[end + 1:].strip().startswith('#')):
                raise ValueError(f'Invalid .env quoting at line {number}.')
            value = value[1:end]
        else:
            value = '' if value.startswith('#') else re.split(r'\s+#', value, maxsplit=1)[0].rstrip()
        # Blank template credentials leave saved `hf auth login` credentials usable.
        if value:
            values[key] = value
    return values


def environment_values(path: Path | None = None, *, environ=None) -> dict[str, str]:
    """Return a fresh child environment; explicit shell values override the file."""
    values = dotenv_values(path)
    values.update(dict(os.environ if environ is None else environ))
    return values


def load_environment(path: Path | None = None) -> None:
    """Load before config/Hub/CUDA imports; do not replace existing shell settings."""
    for key, value in dotenv_values(path).items():
        os.environ.setdefault(key, value)


def credential_status(path: Path | None = None) -> dict[str, str]:
    """Status only, safe to show in notebook outputs and logs."""
    values = environment_values(path)
    return {key: 'set' if values.get(key, '').strip() else 'empty' for key in CREDENTIAL_KEYS}


def redact(text: str, secrets) -> str:
    for value in sorted(set(secrets), key=len, reverse=True):
        if value:
            text = text.replace(value, '[REDACTED]')
    return text
