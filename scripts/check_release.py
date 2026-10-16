"""Build and exercise an installed wheel without importing the working checkout."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(arguments: list[str], *, cwd: Path, env: dict, capture: bool = False,
        expected: int = 0) -> str:
    result = subprocess.run(arguments, cwd=cwd, env=env, text=True,
                            capture_output=capture, timeout=300)
    if result.returncode != expected:
        if capture:
            print(result.stdout, end="", file=sys.stderr)
            print(result.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"{Path(arguments[0]).name} returned {result.returncode}; expected {expected}")
    return result.stdout if capture else ""


def check_release() -> None:
    require(importlib.util.find_spec("build") is not None,
            "Install the build tool first: python -m pip install build")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    with TemporaryDirectory(prefix="llm-trace-release-") as directory:
        temporary = Path(directory)
        source = temporary / "source checkout"
        source.mkdir()
        # Build fresh inputs, excluding stale builds, environments, and local databases.
        for name in ("pyproject.toml", "README.md", "MANIFEST.in"):
            shutil.copy2(ROOT / name, source / name)
        for name in ("src/llm_trace", "tests", "examples", "scripts", "docs"):
            shutil.copytree(ROOT / name, source / name,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", "*.pyc", "*.egg-info", "*.sqlite3", "*.sqlite3-*",
                                ".env", ".env.*", ".venv", ".release-env", ".git", "build", "dist"))
        artifacts = temporary / "artifacts"
        print("Building a source distribution, then a wheel from that distribution.", flush=True)
        # Without --sdist/--wheel, build checks the source distribution by building its wheel.
        run([sys.executable, "-I", "-m", "build", str(source), "--outdir", str(artifacts)],
            cwd=temporary, env=env)
        wheels = list(artifacts.glob("*.whl"))
        require(len(wheels) == 1 and len(list(artifacts.glob("*.tar.gz"))) == 1,
                "Build must produce exactly one wheel and one source distribution")

        environment = temporary / "clean environment"
        run([sys.executable, "-I", "-m", "venv", str(environment)], cwd=temporary, env=env)
        binaries = environment / ("Scripts" if os.name == "nt" else "bin")
        python = binaries / ("python.exe" if os.name == "nt" else "python")
        executable = binaries / ("llm-trace.exe" if os.name == "nt" else "llm-trace")
        run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(wheels[0])],
            cwd=temporary, env=env)
        probe = run([str(python), "-I", "-c", """
import importlib.util
from importlib.metadata import distribution
import json
import llm_trace
package = distribution('llm-trace-local')
print(json.dumps({'version': package.version, 'requires': package.requires or [],
                  'path': llm_trace.__file__, 'openai_installed': importlib.util.find_spec('openai') is not None}))
"""], cwd=temporary, env=env, capture=True)
        installed = json.loads(probe)
        require(not installed["requires"], "The base wheel declares runtime dependencies")
        require(not installed["openai_installed"], "The clean environment unexpectedly includes OpenAI")
        require(Path(installed["path"]).resolve().is_relative_to(environment.resolve()),
                "Imported package did not come from the clean environment")
        require(executable.is_file(), "Installed command-line entry point is missing")

        print("Running tests against the installed package, outside the source checkout.", flush=True)
        run([str(python), "-I", "-m", "unittest", "discover", "-s", str(source / "tests"), "-v"],
            cwd=temporary, env=env)
        db = temporary / "demo.sqlite3"

        def cli(*arguments: str, database: Path = db, expected: int = 0) -> str:
            return run([str(executable), "--db", str(database), *arguments],
                       cwd=temporary, env=env, capture=True, expected=expected)

        cli("--help")
        cli("demo")
        listing = cli("list")
        ids = [line.split()[0] for line in listing.splitlines()]
        require(len(ids) == 4, "Installed demo did not save four traces")
        traces = [json.loads(cli("export", trace_id)) for trace_id in ids]
        by_name = {trace["name"]: trace["id"] for trace in traces}
        require(set(by_name) == {"normal", "slow-generation", "recovered-lookup", "failed-lookup"},
                "Installed demo cases changed")
        cli("show", by_name["normal"][:8])
        summary = json.loads(cli("summary", by_name["recovered-lookup"], "--json"))
        require(summary["status"] == "ok" and len(summary["failed_spans"]) == 1,
                "Recovered failure is missing from the installed summary")
        comparison = json.loads(cli("compare", by_name["normal"], by_name["slow-generation"], "--json"))
        require(bool(comparison["calls"]), "Installed comparison contains no calls")
        report = temporary / "report.html"
        cli("report", by_name["normal"], "--output", str(report))
        original = report.read_bytes()
        require(original.startswith(b"<!doctype html>"), "Installed report is not an HTML document")
        cli("report", by_name["normal"], "--output", str(report), expected=1)
        require(report.read_bytes() == original, "Report command overwrote an existing file")
        missing = temporary / "missing.sqlite3"
        cli("list", database=missing, expected=1)
        require(not missing.exists(), "Read command created a missing database")
        benchmark = json.loads(cli("benchmark", "--rounds", "3", "--warmup", "0",
                                   "--spans", "1", "--work", "0", "--json"))
        require(benchmark["saved_traces"] == 3 and benchmark["checksum"] == 1,
                "Installed benchmark did not verify its saved workload")
        print(f"Release check passed: llm-trace-local {installed['version']}, Python {sys.version.split()[0]}.")
        print("Source distribution, installed wheel, tests, and all CLI commands checked; no model calls.")


def main() -> int:
    try:
        check_release()
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Release check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
