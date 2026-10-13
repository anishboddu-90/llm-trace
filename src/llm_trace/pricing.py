"""Token-cost arithmetic using caller-supplied USD rates, not a pricing catalog."""

from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext
import math

Rate = Decimal | int | float | str


def _rate(value: Rate) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, float, str)):
        raise ValueError("rates must be finite, non-negative decimal values")
    try:
        rate = Decimal(str(value))
    except (DecimalException, ValueError):
        raise ValueError("rates must be finite, non-negative decimal values") from None
    if not rate.is_finite() or rate < 0:
        raise ValueError("rates must be finite, non-negative decimal values")
    return rate


@dataclass(frozen=True)
class TokenPricing:
    input_per_million: Rate
    output_per_million: Rate
    cached_input_per_million: Rate | None = None

    def __post_init__(self):
        for name in ("input_per_million", "output_per_million", "cached_input_per_million"):
            value = getattr(self, name)
            if value is not None or name != "cached_input_per_million":
                object.__setattr__(self, name, _rate(value))

    def to_dict(self) -> dict[str, str | None]:
        return {name: str(value) if value is not None else None for name, value in (
            ("input_per_million", self.input_per_million),
            ("output_per_million", self.output_per_million),
            ("cached_input_per_million", self.cached_input_per_million),
        )}

    def estimate(self, *, input_tokens: int, output_tokens: int,
                 cached_input_tokens: int | None = None) -> float | None:
        for count in (input_tokens, output_tokens):
            if type(count) is not int or count < 0:
                raise ValueError("token counts must be non-negative integers")
        if cached_input_tokens is not None:
            if type(cached_input_tokens) is not int or not 0 <= cached_input_tokens <= input_tokens:
                raise ValueError("cached input tokens must be between zero and input tokens")
        if self.cached_input_per_million is not None and cached_input_tokens is None:
            return None  # A discount can't be estimated without the cache count.
        cached = cached_input_tokens if self.cached_input_per_million is not None else 0
        try:
            with localcontext() as context:
                context.prec = 40
                amount = ((input_tokens - cached) * self.input_per_million
                          + cached * (self.cached_input_per_million or Decimal(0))
                          + output_tokens * self.output_per_million) / Decimal(1_000_000)
                result = float(amount)
        except (DecimalException, OverflowError):
            return None
        if not math.isfinite(result) or (result == 0 and amount != 0):
            return None
        return result
