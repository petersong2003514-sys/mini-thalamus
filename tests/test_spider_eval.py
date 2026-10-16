import json
import os
import sqlite3
import sys
import tempfile
import unittest
import zipfile



import mini_thalamus.evals.spider_eval as spider_eval  # noqa: E402


class SpiderEvalTests(unittest.TestCase):
    def test_fetch_rows_and_execution_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.sqlite")
            connection = sqlite3.connect(path)
            connection.execute("CREATE TABLE people (id INTEGER, name TEXT)")
            connection.executemany(
                "INSERT INTO people VALUES (?, ?)",
                [(1, "alice"), (2, "bob")],
            )
            connection.commit()
            connection.close()

            columns, rows = spider_eval.fetch_rows(
                path,
                "SELECT name FROM people ORDER BY id",
            )
            self.assertEqual(columns, ["name"])
            self.assertEqual(rows, [("alice",), ("bob",)])
            self.assertTrue(
                spider_eval.execution_match(rows, [("bob",), ("alice",)])
            )

    def test_download_spider_accepts_local_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = os.path.join(tmp, "spider.zip")
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("spider/dev.json", json.dumps([]))
                archive.writestr("spider/database/x/x.sqlite", b"")
            spider_eval.download_spider(
                data_dir=os.path.join(tmp, "out"),
                zip_url=zip_path,
            )
            self.assertTrue(
                os.path.exists(os.path.join(tmp, "out", "spider", "dev.json"))
            )


if __name__ == "__main__":
    unittest.main()
