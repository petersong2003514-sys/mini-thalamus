import unittest


class FullEvidenceCommandTests(unittest.TestCase):
    def test_build_full_commands_covers_multi_turn_bfcl_fc(self):
        from mini_thalamus.evals.scripts.run_full_evidence import build_full_commands

        cmds = build_full_commands(
            ["simple_python", "parallel", "multi_turn_base", "web_search"]
        )
        self.assertTrue(
            any(
                "--mode" in cmd
                and "bfcl-fc" in cmd
                and any("multi_turn" in str(part) for part in cmd)
                for cmd in cmds
            )
        )
        self.assertTrue(any("evals/spider_eval.py" in cmd for cmd in cmds))
        self.assertTrue(any("evals/beir_eval.py" in cmd for cmd in cmds))


if __name__ == "__main__":
    unittest.main()
