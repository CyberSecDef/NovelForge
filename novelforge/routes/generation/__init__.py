"""Chapter generation, revision, and progress routes.

The blueprint and shared constants live in ``_shared``; route handlers are
spread across ``chapters`` (generation + progress), ``revision``, and
``audits``.  Importing this package triggers the route registrations via
the submodule imports below.

All symbols previously importable from ``novelforge.routes.generation`` are
re-exported here so that existing call sites and test fixtures work unchanged.
"""

# -- Shared state (blueprint, constants) -------------------------------------

from novelforge.routes.generation._shared import (  # noqa: F401
    generation_bp,
    _PROGRESS_PERSIST_INTERVAL,
    _DERIVED_REPORT_FIELDS,
)

# -- LLM imports (kept at package level so test fixtures can patch them) -----

from novelforge.llm.client import call_llm, parse_llm_json, friendly_llm_error  # noqa: F401

# -- Route submodules (import triggers @generation_bp.route registration) ----

from novelforge.routes.generation import chapters as _chapters  # noqa: F401
from novelforge.routes.generation import revision as _revision  # noqa: F401

# -- stdlib re-exports (tests patch gen_mod.threading.Thread) ----------------

import threading  # noqa: F401

# -- Re-exports for test compatibility --------------------------------------

from novelforge.routes.generation.chapters import (  # noqa: F401
    _run_chapter_generation,
    _run_chapter_generation_internal,
    _resume_chapter_generation,
    _LIGHTWEIGHT_FIELDS,
)
from novelforge.routes.generation.audits import (  # noqa: F401
    run_post_manuscript_audits,
)
