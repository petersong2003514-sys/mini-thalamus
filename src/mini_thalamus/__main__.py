"""`python -m mini_thalamus` starts the web app.

Use `python -m mini_thalamus.main` (or the `mini-thalamus-cli` script) for the
terminal CLI instead.
"""

from mini_thalamus.web_server import main

if __name__ == "__main__":
    main()
