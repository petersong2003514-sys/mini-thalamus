import json
import os
import sys
import tempfile
import unittest


EVALS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evals",
)
sys.path.insert(0, EVALS_DIR)

import beir_eval  # noqa: E402


class BeirEvalTests(unittest.TestCase):
    def test_smoke_fixture_loads_and_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = beir_eval.write_smoke_fixture(tmp)
            corpus, queries, qrels = beir_eval.load_dataset_dir(
                "_smoke",
                split="test",
                data_dir=tmp,
            )
            self.assertEqual(len(corpus), 3)
            self.assertEqual(len(queries), 2)
            self.assertEqual(qrels["q1"], {"d1"})
            metrics, records = beir_eval.evaluate(
                corpus,
                queries,
                qrels,
                top_k=10,
                bm25=True,
                encoder="unused",
            )
            self.assertEqual(metrics["case_count"], 2)
            self.assertIn("ndcg_at_10", metrics)
            self.assertIn("mrr_at_10", metrics)
            self.assertEqual(len(records), 2)

    def test_metrics_without_qrels_are_zero(self):
        corpus = {"d1": "alpha beta", "d2": "gamma delta"}
        queries = {"q1": "alpha"}
        metrics, _ = beir_eval.evaluate(
            corpus,
            queries,
            {},
            top_k=10,
            bm25=True,
            encoder="unused",
        )
        self.assertEqual(metrics["mrr_at_10"], 0)
        self.assertEqual(metrics["ndcg_at_10"], 0)


if __name__ == "__main__":
    unittest.main()
