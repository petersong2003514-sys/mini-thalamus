"""Mini Thalamus — a router-first multi-agent platform.

The thalamus is the brain's relay station: it receives signals from
everywhere, decides which cortical region each one belongs to, and gates
what is allowed to pass through. That is this project's job too — a
cascaded router (rules -> semantics -> LLM) dispatches each request to one
of the specialist agents, and an approval checkpoint gates the operations
that are allowed to touch the outside world.

Public entry points:

    python -m mini_thalamus.web_server     # the web app (http://127.0.0.1:8765)
    python -m mini_thalamus.main           # the CLI
"""

import os

#: Directory holding the packaged data: skills/, data/, static/, evals/.
#: Bundled assets are located relative to this rather than to the caller's
#: working directory, so the app behaves the same however it was started.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

__version__ = "0.1.0"
