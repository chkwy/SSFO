import json


def read_json(file_path):
    """
    Reads a JSON file and returns the data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(data, file_path):
    """
    Writes data to a JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
