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
2. `pixi install -e dev`


# Release
For releasing draft a new release on https://github.com/HHNK/hhnk-research-tools/releases.

The naming should equal "v" + the version in pyproject.toml; e.g. `v2025.1.0`

A secret has been configured in https://github.com/HHNK/hhnk-research-tools/settings/secrets/actions.\
When this release is published it will runn the gh action publish_on_release.yml.
This runs:\
`pixi run build`\
`pixi run test`\
`pixi run twine_upload_test`
