"""
Tests verifying that .env loading is consistent across all entry points.

The fix moves load_dotenv() from app.py into novelforge/config.py so that
every supported startup path – direct script, WSGI/ASGI server, and direct
create_app() imports – benefits from the same environment bootstrap.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


# Repository root (one level above this tests/ directory)
_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestAppPyIsThinLauncher:
    """app.py must not contain any unique configuration side effects."""

    def test_app_py_does_not_call_load_dotenv(self):
        """app.py must not call load_dotenv() after the fix."""
        source = (_REPO_ROOT / "app.py").read_text()
        assert "load_dotenv" not in source, (
            "app.py still calls load_dotenv(). "
            "Environment loading must live in novelforge/config.py."
        )

    def test_app_py_does_not_import_dotenv(self):
        """app.py must not import from dotenv after the fix."""
        source = (_REPO_ROOT / "app.py").read_text()
        assert "from dotenv" not in source, (
            "app.py still imports from the dotenv package. "
            "That import must live in novelforge/config.py."
        )


class TestConfigOwnsEnvLoading:
    """novelforge/config.py must own the load_dotenv() call."""

    def test_config_calls_load_dotenv(self):
        """config.py must contain the load_dotenv() call."""
        source = (_REPO_ROOT / "novelforge" / "config.py").read_text()
        assert "load_dotenv" in source, (
            "novelforge/config.py does not call load_dotenv(). "
            "Environment loading must be centralised there."
        )

    def test_config_imports_load_dotenv_from_dotenv(self):
        """config.py must import load_dotenv from the dotenv package."""
        source = (_REPO_ROOT / "novelforge" / "config.py").read_text()
        assert "from dotenv import load_dotenv" in source, (
            "novelforge/config.py does not import load_dotenv from dotenv."
        )

    def test_load_dotenv_called_before_first_env_read(self):
        """load_dotenv() must appear before the first os.environ.get call."""
        source = (_REPO_ROOT / "novelforge" / "config.py").read_text()
        load_dotenv_pos = source.find("load_dotenv(")
        first_env_get_pos = source.find("os.environ.get(")
        assert load_dotenv_pos != -1, "load_dotenv() call not found in config.py"
        assert first_env_get_pos != -1, "os.environ.get() call not found in config.py"
        assert load_dotenv_pos < first_env_get_pos, (
            "load_dotenv() must appear before the first os.environ.get() call "
            "so that .env values are available when module-level variables are set."
        )


class TestConsistentEnvLoadingAcrossEntrypoints:
    """
    End-to-end verification that .env is loaded regardless of how the
    application is started.

    Each test spins up a fresh subprocess so that the module import cache is
    clean and we can write a real .env file that load_dotenv() picks up.
    """

    def _run_script(self, script: str, env: dict | None = None) -> subprocess.CompletedProcess:
        """Run *script* in a subprocess with the repo root on PYTHONPATH."""
        subprocess_env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
        if env:
            subprocess_env.update(env)
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=subprocess_env,
        )

    def test_create_app_loads_dotenv_without_app_py(self, tmp_path):  # noqa: ARG002
        """
        Importing and calling create_app() directly (bypassing app.py) must
        pick up values from a .env file in the project root.

        The test temporarily appends a unique variable to the project-root
        .env file, spawns a subprocess that imports the package (triggering
        load_dotenv() in config.py) without going through app.py, and
        verifies that the variable was loaded into the environment.
        """

        env_file = _REPO_ROOT / ".env"
        var_name = "NF_DOTENV_ENTRYPOINT_TEST"
        original = env_file.read_text() if env_file.exists() else None

        # Append the test variable to the real .env (or create a minimal one)
        try:
            existing = original or ""
            if var_name not in existing:
                env_file.write_text(existing + f"\n{var_name}=from_dotenv_file\n")

            script = textwrap.dedent(f"""\
                import os
                # Ensure the variable is not inherited from the parent process.
                os.environ.pop({var_name!r}, None)
                # Import the package – this triggers load_dotenv() in config.py.
                from novelforge import create_app  # noqa: F401
                val = os.environ.get({var_name!r}, "NOT_LOADED")
                print(val, end="")
            """)

            result = self._run_script(script)
            assert result.returncode == 0, (
                f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            assert result.stdout == "from_dotenv_file", (
                f"Expected 'from_dotenv_file', got {result.stdout!r}. "
                "create_app() import did not load the .env file."
            )
        finally:
            # Restore the .env file to its original state.
            if original is None:
                env_file.unlink(missing_ok=True)
            else:
                env_file.write_text(original)

    def test_dotenv_does_not_override_existing_env_vars(self, tmp_path):  # noqa: ARG002
        """
        Variables already set in the process environment must not be
        overridden by .env values (override=False is the correct behaviour).
        """
        env_file = _REPO_ROOT / ".env"
        var_name = "NF_DOTENV_OVERRIDE_TEST"
        original = env_file.read_text() if env_file.exists() else None

        try:
            existing = original or ""
            if var_name not in existing:
                env_file.write_text(existing + f"\n{var_name}=from_dotenv_file\n")

            script = textwrap.dedent(f"""\
                import os
                # Pre-set the variable BEFORE the package is imported.
                os.environ[{var_name!r}] = "from_environment"
                from novelforge import create_app  # noqa: F401
                val = os.environ.get({var_name!r}, "NOT_SET")
                print(val, end="")
            """)

            result = self._run_script(script)
            assert result.returncode == 0, (
                f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
            assert result.stdout == "from_environment", (
                f"Expected 'from_environment', got {result.stdout!r}. "
                "load_dotenv() must not override an existing env var "
                "(override=False should be respected)."
            )
        finally:
            if original is None:
                env_file.unlink(missing_ok=True)
            else:
                env_file.write_text(original)
