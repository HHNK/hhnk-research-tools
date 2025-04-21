import json
from pathlib import Path


def read_api_file(file_path):
    """Read Lizard and 3Di api-keys from a JSON-file. The JSON-file should contain e.g.:
    {"lizard: "valid_lizard_key","threedi: "valid_threedi_key"}
    """
    api_keys = {"lizard": "", "threedi": ""}
    file_path = Path(file_path)
    if file_path.exists():
        try:
            result = json.loads(file_path.read_text())
            if isinstance(result, dict):
                for k in api_keys.keys():
                    if k in result.keys():
                        api_keys[k] = result[k]

        except Exception as e:
            raise e
    return api_keys
