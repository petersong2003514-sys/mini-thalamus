import unittest

from mini_thalamus.tools.url_fetcher import extract_text


class ExtractTextTests(unittest.TestCase):
    def test_void_tags_do_not_skip_body_text(self):
        html = (
            "<html><head><meta charset=\"utf-8\"><title>Page</title>"
            "<link rel=\"stylesheet\" href=\"style.css\"></head>"
            "<body><p>Hello world</p></body></html>"
        )
        self.assertIn("Hello world", extract_text(html))

    def test_script_content_is_skipped_and_body_survives(self):
        html = (
            "<html><head><meta charset=\"utf-8\"></head><body>"
            "<script>var x = 1;</script>"
            "<p>Visible text</p>"
            "</body></html>"
        )
        text = extract_text(html)
        self.assertIn("Visible text", text)
        self.assertNotIn("var x", text)

    def test_many_void_tags_do_not_accumulate_skip_depth(self):
        html = (
            "<html><head>"
            "<meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width\">"
            "<link rel=\"icon\" href=\"favicon.ico\">"
            "<link rel=\"stylesheet\" href=\"a.css\">"
            "</head><body><p>Final answer</p></body></html>"
        )
        self.assertIn("Final answer", extract_text(html))


if __name__ == "__main__":
    unittest.main()
