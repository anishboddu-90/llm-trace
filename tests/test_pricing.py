from decimal import Decimal
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from llm_trace import TokenPricing, Tracer
from llm_trace.integrations.openai import OpenAIResponses


class PricingTests(unittest.TestCase):
    def test_uniform_rates_and_zero_usage(self):
        rates = TokenPricing(input_per_million="1.25", output_per_million="5")
        self.assertEqual(rates.estimate(input_tokens=800, output_tokens=200), 0.002)
        self.assertEqual(rates.estimate(input_tokens=0, output_tokens=0), 0.0)
        self.assertEqual(rates.estimate(input_tokens=800, output_tokens=200, cached_input_tokens=400), 0.002)

    def test_cached_rate_does_not_double_count_input(self):
        rates = TokenPricing(input_per_million=2, output_per_million=10, cached_input_per_million="0.5")
        self.assertEqual(rates.estimate(input_tokens=1000, output_tokens=100, cached_input_tokens=800), 0.0018)
        self.assertIsNone(rates.estimate(input_tokens=1000, output_tokens=100))
        free_cache = TokenPricing(2, 10, 0)
        self.assertEqual(free_cache.estimate(input_tokens=1000, output_tokens=100, cached_input_tokens=1000), 0.001)

    def test_snapshot_is_json_compatible_and_rates_are_immutable(self):
        rates = TokenPricing(Decimal("1.25"), 5, "0.5")
        self.assertEqual(rates.to_dict(), {"input_per_million": "1.25", "output_per_million": "5", "cached_input_per_million": "0.5"})
        with self.assertRaises(AttributeError):
            rates.input_per_million = 100

    def test_invalid_rates_and_counts_are_rejected(self):
        for value in (-1, True, None, "NaN", "Infinity", "bad", object()):
            for position in ("input", "output", "cached"):
                arguments = {"input_per_million": 1, "output_per_million": 2}
                arguments[{"input": "input_per_million", "output": "output_per_million", "cached": "cached_input_per_million"}[position]] = value
                if position == "cached" and value is None:
                    continue
                with self.subTest(position=position, value=value), self.assertRaises(ValueError):
                    TokenPricing(**arguments)
        rates = TokenPricing(1, 2)
        for count in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                rates.estimate(input_tokens=count, output_tokens=1)
        for count in (-1, True, 11):
            with self.assertRaises(ValueError):
                rates.estimate(input_tokens=10, output_tokens=1, cached_input_tokens=count)

    def test_unrepresentable_cost_stays_unknown(self):
        self.assertIsNone(TokenPricing("1e1000", 1).estimate(input_tokens=1, output_tokens=1))
        self.assertIsNone(TokenPricing("1e-1000", 0).estimate(input_tokens=1, output_tokens=0))


class AdapterPricingTests(unittest.TestCase):
    def run_response(self, *, cached, pricing):
        result = SimpleNamespace(model="fixture", status="completed", usage=SimpleNamespace(
            input_tokens=1000, output_tokens=100, input_tokens_details=SimpleNamespace(cached_tokens=cached),
        ))
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=result)))
        tracer = Tracer()
        with tracer.trace("request") as trace:
            self.assertIs(OpenAIResponses(tracer, client, pricing=pricing).create(model="fixture"), result)
        return trace.spans[0]

    def test_adapter_records_cost_cache_count_and_pricing_basis(self):
        pricing = TokenPricing(2, 10, "0.5")
        span = self.run_response(cached=800, pricing=pricing)
        self.assertEqual(span.cached_input_tokens, 800)
        self.assertEqual(span.estimated_cost_usd, 0.0018)
        self.assertEqual(span.token_pricing, pricing.to_dict())

    def test_invalid_or_missing_cache_data_does_not_guess_discount(self):
        for cached in (None, True, -1, 1001):
            with self.subTest(cached=cached):
                span = self.run_response(cached=cached, pricing=TokenPricing(2, 10, "0.5"))
                self.assertIsNone(span.cached_input_tokens)
                self.assertIsNone(span.estimated_cost_usd)
                self.assertEqual(span.input_tokens, 1000)
                self.assertEqual(span.status, "ok")

    def test_no_pricing_leaves_cost_unknown_but_keeps_cache_usage(self):
        span = self.run_response(cached=800, pricing=None)
        self.assertEqual(span.cached_input_tokens, 800)
        self.assertIsNone(span.estimated_cost_usd)
        self.assertIsNone(span.token_pricing)

    def test_wrong_pricing_type_is_rejected_before_a_call(self):
        with self.assertRaises(TypeError):
            OpenAIResponses(Tracer(), Mock(), pricing={"input": 1})

    def test_manual_span_rejects_invalid_cache_count_without_partial_update(self):
        tracer = Tracer()
        with tracer.trace("request"):
            with tracer.span("generate", kind="model") as span:
                span.set_usage(input_tokens=10, output_tokens=2, cached_input_tokens=4)
                for value in (-1, True, 21):
                    with self.assertRaises(ValueError):
                        span.set_usage(input_tokens=20, output_tokens=3, cached_input_tokens=value)
                self.assertEqual((span.input_tokens, span.cached_input_tokens), (10, 4))
