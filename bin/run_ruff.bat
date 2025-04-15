REM force-exclude to make sure it uses the extend-exclude from pyproject.
python -m ruff check ../tests_hrt --select I --fix 
python -m ruff format ../tests_hrt/**/*.py --force-exclude
python -m ruff check ../hhnk_research_tools --select I --fix 
python -m ruff format ../hhnk_research_tools/**/*.py --force-exclude