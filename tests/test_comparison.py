import copy
import unittest

from llm_trace import Tracer
from llm_trace.comparison import compare_traces


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        tracer = Tracer()
        with tracer.trace("baseline") as trace:
            for name in ("left", "right"):
                with tracer.span(name):
                    with tracer.span("generate", kind="model", model="model-a", prompt_version="v1") as call:
                        call.set_usage(input_tokens=10, output_tokens=4, estimated_cost_usd=0.002)
        data = trace.to_dict()
        data["duration_ms"] = 100
        for span in data["spans"]:
            span["duration_ms"] = 10
        return data

    def test_matching_uses_parent_paths_and_reports_changed_metadata(self):
        base = self.fixture()
        new = copy.deepcopy(base)
        new["name"], new["id"], new["duration_ms"] = "candidate", "c" * 32, 150
        new["spans"][1].update(duration_ms=30, model="model-b", prompt_version="v2")
        result = compare_traces(base, new)
        self.assertEqual(result["duration_ms"]["delta"], 50)
        self.assertEqual(result["duration_ms"]["percent_change"], 50)
        model_groups = [group for group in result["calls"] if group["path"][-1]["kind"] == "model"]
        self.assertEqual(len(model_groups), 2)
        left = next(group for group in model_groups if group["path"][0]["name"] == "left")
        right = next(group for group in model_groups if group["path"][0]["name"] == "right")
        self.assertEqual(left["mean_duration_ms"]["delta"], 20)
        self.assertEqual(left["candidate"]["prompt_versions"], ["v2"])
        self.assertEqual(left["candidate"]["models"], ["model-b"])
        self.assertEqual(right["mean_duration_ms"]["delta"], 0)

    def test_repeated_calls_compare_means_without_hiding_count_changes(self):
        base = self.fixture()
        new = copy.deepcopy(base)
        duplicate = copy.deepcopy(new["spans"][1])
        duplicate.update(id="d" * 32, duration_ms=30)
        new["spans"].append(duplicate)
        result = compare_traces(base, new)
        left = next(group for group in result["calls"] if len(group["path"]) == 2 and group["path"][0]["name"] == "left")
        self.assertEqual(left["baseline"]["count"], 1)
        self.assertEqual(left["candidate"]["count"], 2)
        self.assertEqual(left["mean_duration_ms"]["candidate"], 20)
        self.assertEqual(left["mean_duration_ms"]["delta"], 10)

    def test_added_removed_and_failed_calls(self):
        base = self.fixture()
        new = copy.deepcopy(base)
        new["spans"][1].update(name="different-call", status="error", error_type="TimeoutError")
        result = compare_traces(base, new)
        self.assertEqual(result["failed_spans"]["delta"], 1)
        self.assertEqual({group["presence"] for group in result["calls"]}, {"matched", "added", "removed"})
        for group in result["calls"]:
            if group["presence"] != "matched":
                self.assertIsNone(group["mean_duration_ms"]["delta"])

    def test_partial_usage_and_cost_are_not_compared_as_complete_totals(self):
        base = self.fixture()
        new = copy.deepcopy(base)
        new["spans"][1].update(input_tokens=None, output_tokens=None, estimated_cost_usd=None)
        result = compare_traces(base, new)
        for field in ("input_tokens", "output_tokens", "estimated_cost_usd"):
            self.assertIsNone(result[field]["candidate"])
            self.assertIsNone(result[field]["delta"])
        self.assertEqual(result["input_tokens"]["baseline"], 20)

    def test_zero_baseline_has_no_percentage_and_empty_trace_has_zero_model_usage(self):
        tracer = Tracer()
        with tracer.trace("empty") as trace:
            pass
        base = trace.to_dict()
        base["duration_ms"] = 0
        new = copy.deepcopy(base)
        new["duration_ms"] = 10
        result = compare_traces(base, new)
        self.assertEqual(result["duration_ms"]["delta"], 10)
        self.assertIsNone(result["duration_ms"]["percent_change"])
        self.assertEqual(result["input_tokens"]["delta"], 0)
        self.assertEqual(result["calls"], [])

    def test_invalid_parent_graphs_fail_clearly(self):
        base = self.fixture()
        for mode in ("missing", "cycle", "duplicate"):
            new = copy.deepcopy(base)
            if mode == "missing":
                new["spans"][0]["parent_id"] = "missing"
            elif mode == "cycle":
                new["spans"][0]["parent_id"] = new["spans"][1]["id"]
            else:
                new["spans"].append(copy.deepcopy(new["spans"][0]))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                compare_traces(base, new)
