import io
import unittest
from unittest.mock import patch

from PIL import Image

import mini_thalamus.agents.vision_agent as vision_agent


class FakeOCRReader:
    def ocr(self, path):
        return [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "Hello OCR", 0.99),
        ]


class VisionOCRWrapperTests(unittest.TestCase):
    def test_ocr_image_returns_detected_text(self):
        buf = io.BytesIO()
        Image.new("RGB", (40, 20), "white").save(buf, format="PNG")
        with patch.object(vision_agent, "_ocr_reader", FakeOCRReader()):
            text = vision_agent._ocr_image(buf.getvalue())

        self.assertIn("Hello OCR", text)


if __name__ == "__main__":
    unittest.main()
