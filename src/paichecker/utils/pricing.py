from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPricing:
    input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


MODEL_PRICING: dict[str, TokenPricing] = {
    "gpt": TokenPricing(input_per_1m=1.75, cached_input_per_1m=1.75, output_per_1m=14.0),
    "claude": TokenPricing(input_per_1m=3.0, cached_input_per_1m=3, output_per_1m=15.0),
    "gemini": TokenPricing(input_per_1m=2.0, cached_input_per_1m=2, output_per_1m=12.0),
    "qwen": TokenPricing(input_per_1m=0.8, cached_input_per_1m=0.8, output_per_1m=4.8),
}


def resolve_model_key(model_name: str | None) -> str:
    if not model_name:
        return "gpt"
    name = model_name.lower()
    for key in ("claude", "gemini", "qwen", "gpt"):
        if key in name:
            return key
    return "gpt"


def get_token_pricing(model_name: str | None) -> TokenPricing:
    return MODEL_PRICING[resolve_model_key(model_name)]


def estimate_cost_usd(
    *,
    prompt_tokens: int,
    cached_input_tokens: int,
    completion_tokens: int,
    model_name: str | None,
) -> tuple[float, int, TokenPricing]:
    pricing = get_token_pricing(model_name)
    non_cached_input_tokens = max(prompt_tokens - cached_input_tokens, 0)
    cost = (
        (non_cached_input_tokens / 1_000_000) * pricing.input_per_1m
        + (cached_input_tokens / 1_000_000) * pricing.cached_input_per_1m
        + (completion_tokens / 1_000_000) * pricing.output_per_1m
    )
    return cost, non_cached_input_tokens, pricing
