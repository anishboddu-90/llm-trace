# Check an install before release

From a checkout, using Python 3.11 or newer:

```sh
python -m venv .release-env
source .release-env/bin/activate
python -m pip install build
python scripts/check_release.py
```

On Windows, activate with `.release-env\Scripts\activate` instead. Use the Python version you intend to check. The command reports that version when it succeeds; one run doesn't establish compatibility with other Python versions or operating systems.

`build` is a development tool and isn't a dependency of llm-trace. The build step may download its backend requirements. Installation of the resulting wheel uses `--no-index --no-deps`, so it cannot fetch missing runtime packages to hide a packaging mistake. Tests and command checks make no model calls and need no API keys.

The script copies the source, tests, examples, and documentation into a temporary directory. It excludes existing build output, local databases, and environments. It builds a source archive and then builds the wheel from that archive. This catches package files missing from the source distribution.

Next it creates a fresh environment, installs only that wheel, and checks that the imported package comes from that environment, has no runtime dependencies, and works without the OpenAI SDK. It removes `PYTHONPATH` and `PYTHONHOME` from child processes and runs Python in isolated mode for the package probe and test discovery. The tests run from outside the source checkout against the installed package. The three optional SDK contract tests are skipped in this environment; run them separately with the SDK installed.

Finally it exercises the installed `llm-trace` entry point: help, demo, list, show, export, summary, comparison, HTML report, and a small benchmark. It checks that a report cannot overwrite an existing file and that a read command doesn't create a missing database. All generated files and the temporary environment are removed on success or failure. The working checkout stays unchanged.

Run this from each supported Python interpreter before tagging a release. This checks the build and installation locally. A trial by someone outside the project and the existing-application case study are still needed. The script doesn't publish a package, create a release, or change repository visibility.
