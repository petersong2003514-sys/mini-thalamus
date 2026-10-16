"""Download the GAIA validation set into evals/external/gaia (D-drive only)."""

import json
import os
import sys
import urllib.request


REPO = "gaia-benchmark/GAIA"
VALIDATION_META = "2023/validation/metadata.jsonl"
ATTACHMENT_PREFIX = "2023/validation/attachments/"
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "external",
    "gaia",
)


def _download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"skip  {os.path.basename(dest)}")
        return False
    print(f"fetch {os.path.basename(dest)}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "mini-thalamus-gaia-eval/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with open(dest, "wb") as handle:
        handle.write(data)
    print(f"  ok  {len(data)} bytes")
    return True


def main():
    meta_path = os.path.join(OUT_DIR, "metadata.jsonl")
    _download(
        f"https://huggingface.co/datasets/{REPO}/resolve/main/{VALIDATION_META}",
        meta_path,
    )

    rows = []
    with open(meta_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    attachment_dir = os.path.join(OUT_DIR, "attachments")
    fetched = 0
    with_attachments = 0
    for row in rows:
        file_name = row.get("file_name") or ""
        if not file_name:
            continue
        with_attachments += 1
        if _download(
            (
                f"https://huggingface.co/datasets/{REPO}/resolve/main/"
                f"{ATTACHMENT_PREFIX}{urllib.request.quote(file_name)}"
            ),
            os.path.join(attachment_dir, os.path.basename(file_name)),
        ):
            fetched += 1

    print(
        json.dumps(
            {
                "total": len(rows),
                "with_attachments": with_attachments,
                "attachments_fetched": fetched,
                "levels": {
                    level: sum(1 for r in rows if r.get("level") == level)
                    for level in sorted({r.get("level") for r in rows})
                },
                "out_dir": OUT_DIR,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
