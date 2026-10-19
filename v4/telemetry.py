"""Buffered step trace, independent of numerical training decisions."""
import csv
from pathlib import Path


class TrainingTrace:
    def __init__(self,model):
        self.path=(Path(model._v4_artifact_dir)/"training_steps.csv") if hasattr(model,"_v4_artifact_dir") else None
        self.rows=[];self.handle=None;self.writer=None
    def __enter__(self):return self
    def append(self,**record):
        self.rows.append(record)
        if len(self.rows)>=64:self.flush()
    def flush(self):
        if not self.rows or self.path is None:
            self.rows.clear();return
        if self.handle is None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.handle=self.path.open("w",newline="",encoding="utf8")
            self.writer=csv.DictWriter(self.handle,fieldnames=list(self.rows[0]))
            self.writer.writeheader()
        self.writer.writerows(self.rows);self.rows.clear();self.handle.flush()
    def __exit__(self,*exc):
        self.flush()
        if self.handle:self.handle.close()
