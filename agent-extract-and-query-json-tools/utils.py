import json
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path


class JsonList:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text())

    @property
    def sample_item(self) -> str:
        return self.data[0]

    def store_variation(self, d: List[Dict[str, Any]]) -> str:
        new_filename = (
            f"{self.path.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        )
        new_path = self.path.parent / new_filename

        new_path.write_text(json.dumps(d))
