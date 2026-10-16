import unittest
from unittest.mock import patch

from agents.docparse_agent import _parse_pdf


class DocParseOCRTests(unittest.TestCase):
    @patch("agents.docparse_agent._try_engine", side_effect=[None, None])
    @patch(
        "agents.docparse_agent._ocr_pdf",
        return_value={
            "paragraphs": [{"page": 1, "text": "扫描页内容"}],
            "tables": [],
            "pages": 1,
        },
    )
    def test_scanned_pdf_falls_back_to_ocr(self, mock_ocr, mock_engine):
        result = _parse_pdf("scan.pdf", "scan.pdf")

        self.assertEqual(result["engine"], "OCR")
        self.assertEqual(result["paragraphs"][0]["text"], "扫描页内容")
        self.assertEqual(mock_ocr.call_count, 1)

    @patch(
        "agents.docparse_agent._try_engine",
        side_effect=[
            {"paragraphs": [{"page": 1, "text": "text pdf"}], "tables": [], "pages": 1},
            None,
        ],
    )
    @patch("agents.docparse_agent._ocr_pdf")
    def test_text_pdf_skips_ocr(self, mock_ocr, mock_engine):
        result = _parse_pdf("text.pdf", "text.pdf")

        self.assertEqual(result["engine"], "pdfplumber")
        mock_ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
