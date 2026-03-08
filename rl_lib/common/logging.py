import csv
from pathlib import Path
from typing import Dict, List


class CSVLogger:
    def __init__(self, path: Path, fieldnames: List[str]):
        self.file = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, row: Dict[str, float]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()
