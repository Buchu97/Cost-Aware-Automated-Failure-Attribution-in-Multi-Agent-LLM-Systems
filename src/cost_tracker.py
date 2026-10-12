"""Compute theoretical commercial API costs from token counts.

These are illustrative API-equivalent costs, not measured local GPU costs
or current commercial price quotes. Local compute still consumes resources.
"""


import config


def compute_theoretical_cost(
    model_alias: str,
    prompt_tokens: int,
    completion_tokens: int,
    model_configs: dict | None = None,
) -> float:
    """Return the theoretical USD cost for one inference call."""
    rates = (model_configs if model_configs is not None else config.MODELS)[model_alias]
    return (
        prompt_tokens * rates["theoretical_cost_per_1m_input"]
        + completion_tokens * rates["theoretical_cost_per_1m_output"]
    ) / 1_000_000


def compute_batch_cost(
    model_alias: str,
    records: list[dict],
    prompt_key: str = "prompt_tokens",
    completion_key: str = "completion_tokens",
) -> float:
    """Sum theoretical cost over a list of result records."""
    return sum(
        compute_theoretical_cost(model_alias, r[prompt_key], r[completion_key])
        for r in records
    )


def compute_usage_cost(settings: dict, record: dict) -> float:
    """Price a single call; reasoning is already included in output tokens."""
    prompt, output = record['prompt_tokens'], record['completion_tokens']
    cached = record.get('cached_prompt_tokens', 0)
    written = record.get('cache_write_prompt_tokens', 0)
    long = prompt > settings.get('long_context_threshold', float('inf'))
    input_multiplier = settings.get('long_context_input_multiplier', 1) if long else 1
    output_multiplier = settings.get('long_context_output_multiplier', 1) if long else 1
    return ((prompt - cached - written + written * settings.get('cache_write_multiplier', 1))
            * settings['theoretical_cost_per_1m_input'] * input_multiplier
            + cached * settings.get('theoretical_cost_per_1m_cached_input',
                                    settings['theoretical_cost_per_1m_input']) * input_multiplier
            + output * settings['theoretical_cost_per_1m_output'] * output_multiplier) / 1_000_000


def aggregate_cost(alias: str, record: dict, model_configs=None) -> float:
    """Use the sum of per-call prices, preserving long-context boundaries."""
    settings = (model_configs if model_configs is not None else config.MODELS)[alias]
    if settings.get('provider') == 'openai':
        from src.run_records import finite_number
        if not finite_number(record.get('usage_cost_usd')):
            raise ValueError(f'Missing API usage price for {alias}; rerun aggregation.')
        return record['usage_cost_usd']
    return compute_theoretical_cost(alias, record.get('prompt_tokens', 0),
                                    record.get('completion_tokens', 0), model_configs)
