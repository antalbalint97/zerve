import unittest

import pandas as pd

from analytics.events import extract_canvas_id, is_build_event, map_country_region, reconstruct_sessions
from analytics.metrics import (
    compute_canvas_complexity_score,
    generate_ngrams,
    label_churn,
    split_first_vs_later_days,
    top_ngram_lift,
    wilson_interval,
)


class AnalyticsHelpersTests(unittest.TestCase):
    def test_extract_canvas_id(self):
        self.assertEqual(
            extract_canvas_id("/canvas/5446a448-57ed-42a4-bb0c-29c6fe264c4e"),
            "5446a448-57ed-42a4-bb0c-29c6fe264c4e",
        )
        self.assertIsNone(extract_canvas_id("/canvases/home"))

    def test_reconstruct_sessions(self):
        df = pd.DataFrame(
            {
                "person_id": ["u1", "u1", "u1"],
                "timestamp": pd.to_datetime(
                    ["2025-01-01 10:00:00", "2025-01-01 10:10:00", "2025-01-01 11:00:00"]
                ),
            }
        )
        out = reconstruct_sessions(df, gap_min=30)
        self.assertEqual(out["derived_session_num"].tolist(), [1, 1, 2])

    def test_compute_canvas_complexity_score(self):
        score = compute_canvas_complexity_score(
            {
                "structural_actions": 3,
                "agent_build_actions": 2,
                "execution_actions": 1,
                "dependency_actions": 0,
                "output_actions": 1,
                "active_days": 2,
            }
        )
        self.assertGreater(score, 0)
        self.assertAlmostEqual(score, 9.8, places=2)

    def test_label_churn(self):
        df = pd.DataFrame(
            {
                "person_id": ["u1", "u2"],
                "timestamp": pd.to_datetime(["2025-01-01", "2025-01-20"]),
            }
        )
        labeled = label_churn(df, horizon_days=14).set_index("person_id")
        self.assertEqual(int(labeled.loc["u1", "is_churn"]), 1)
        self.assertEqual(int(labeled.loc["u2", "is_censored"]), 1)

    def test_generate_ngrams_and_lift(self):
        focal = [["A", "B", "C"], ["A", "B", "D"]]
        baseline = [["A", "C", "D"], ["B", "C", "D"]]
        self.assertEqual(generate_ngrams(["A", "B", "C"], 2), [("A", "B"), ("B", "C")])
        lift = top_ngram_lift(
            focal,
            baseline,
            n=2,
            min_count=1,
            min_baseline_count=1,
            top_k=5,
            smoothing_alpha=1.0,
        )
        self.assertIn("A -> B", set(lift["ngram"]))
        row = lift.set_index("ngram").loc["A -> B"]
        self.assertIn("publishable", lift.columns)
        self.assertGreater(float(row["lift"]), 1.0)

    def test_country_region_mapping(self):
        self.assertEqual(map_country_region("IN"), "India")
        self.assertEqual(map_country_region("US"), "US")
        self.assertEqual(map_country_region("FR"), "EU")

    def test_build_event_filter(self):
        self.assertTrue(is_build_event("agent_tool_call_create_block_tool", ""))
        self.assertTrue(is_build_event("something_else", "refactor_block_tool"))
        self.assertFalse(is_build_event("agent_message", "Coder Agent"))

    def test_split_first_vs_later_days(self):
        df = pd.DataFrame(
            {
                "person_id": ["u1", "u1", "u1", "u1"],
                "canvas_id": ["c1", "c1", "c1", "c1"],
                "timestamp": pd.to_datetime(
                    ["2025-01-01 10:00:00", "2025-01-01 11:00:00", "2025-01-02 10:00:00", "2025-01-03 10:00:00"]
                ),
                "is_structural": [1, 1, 1, 1],
                "is_execution": [0, 1, 1, 0],
                "is_dependency": [0, 0, 0, 1],
                "is_output": [0, 0, 1, 0],
                "is_agent_build": [1, 0, 1, 1],
            }
        )
        out = split_first_vs_later_days(df).set_index(["person_id", "canvas_id"])
        row = out.loc[("u1", "c1")]
        self.assertGreater(float(row["first_day_score"]), 0)
        self.assertEqual(int(row["later_active_days"]), 2)
        self.assertGreaterEqual(float(row["later_max_day_score"]), float(row["later_avg_day_score"]))

    def test_wilson_interval(self):
        low, high = wilson_interval(50, 100)
        self.assertGreaterEqual(low, 0)
        self.assertLessEqual(high, 1)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()
