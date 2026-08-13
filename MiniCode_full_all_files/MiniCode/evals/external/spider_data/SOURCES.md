# Spider dev mirror sources

Official Google Drive link is dead as of 2026-08-13. The dev set is assembled
from two hf-mirror.com datasets:

- Questions: `xlangai/spider` validation parquet
  `https://hf-mirror.com/datasets/xlangai/spider/resolve/main/spider/validation-00000-of-00001.parquet`
- SQLite DBs: `ashishkattamuri/multispider-processed` (covers all 20 dev db_ids)
  `https://hf-mirror.com/datasets/ashishkattamuri/multispider-processed/resolve/main/databases/<db_id>.sqlite`

Local artifacts:

- `spider/dev.json`: 1034 dev cases
- `spider/database/<db_id>/<db_id>.sqlite`: 20 dev databases
- `spider_dev_mirror.zip`: local zip accepted by `spider_eval.py --zip-url`

Verification:

- `python evals/spider_eval.py --download --zip-url evals/external/spider_data/spider_dev_mirror.zip --check-data`
  -> 1034 cases / 20 databases
- First 10 gold queries execute successfully against the mirror SQLite files.
