# HHNK Research Tools

[![pytests](https://github.com/hhnk/hhnk-research-tools/actions/workflows/pytests_research_tools.yml/badge.svg)](https://github.com/hhnk/hhnk-research-tools/actions/workflows/pytests_research_tools.yml)
[![PyPI version](https://badge.fury.io/py/hhnk-research-tools.svg)](https://pypi.org/project/hhnk-research-tools/)
[![Code style](https://img.shields.io/badge/code%20style-ruff-D7FF64)](https://github.com/astral-sh/ruff)
<!-- [![coverage](https://img.shields.io/codecov/c/github/hhnk/hhnk-research-tools)](https://codecov.io/github/hhnk/hhnk-research-tools) -->
---
General tools used in projects across HHNK.

These repo's use this as dependency;

- [https://github.com/threedi/hhnk-threedi-tools](https://github.com/threedi/hhnk-threedi-tools)
- [https://github.com/threedi/hhnk-threedi-plugin](https://github.com/threedi/hhnk-threedi-plugin)
- Hydrologen_projecten
- SPOC
- FEWS

# Installation
1. Install [Pixi](https://pixi.sh/latest/)
2. `pixi install`
3. `pixi run postinstall` -> install pre-commits


# Release
For releasing draft a new release on https://github.com/HHNK/hhnk-research-tools/releases.

The naming should equal "v" + the version in pyproject.toml; e.g. `v2025.1.0`

The [environment](https://github.com/HHNK/hhnk-research-tools/settings/environments) `release` has been created on Github and in it the secret `TESTPYPI_API_TOKEN` has been configured. Get this token from https://test.pypi.org/manage/account/ -> `API tokens`.

When this release is published it will run the gh action publish_on_release.yml.
This runs:\
`pixi run tests`\
`pixi run build_wheels`\
`pixi run twine_check`\
`pixi run twine_upload_test`
