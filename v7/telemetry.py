"""Buffered step trace, independent of numerical training decisions."""
import csv
from pathlib import Path


class TrainingTrace:
    def __init__(self,model):
        self.path=(Path(model._v7_artifact_dir)/"training_steps.csv") if getattr(model,"_v7_artifact_dir",None) else None
        self.rows=[];self.handle=None;self.writer=None;self.resume=bool(getattr(model,"_v7_resume",False));self.recovered=False
    def __enter__(self):return self
    def append(self,**record):
        if not self.recovered and self.path is not None:
            recovery=self.path.parent/"recovery/mean.pt"
            if self.resume and recovery.is_file() and self.path.is_file():
                import torch,pandas as pd
                epoch=torch.load(recovery,map_location="cpu",weights_only=False)["local"]["epoch"]
                old=pd.read_csv(self.path);old[old.epoch<=epoch].to_csv(self.path,index=False)
            self.recovered=True
        self.rows.append(record)
        if len(self.rows)>=64:self.flush()
    def flush(self):
        if not self.rows or self.path is None:
            self.rows.clear();return
        if self.handle is None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            append=self.resume and self.path.is_file() and self.path.stat().st_size>0
            self.handle=self.path.open("a" if append else "w",newline="",encoding="utf8")
            self.writer=csv.DictWriter(self.handle,fieldnames=list(self.rows[0]))
            if not append:self.writer.writeheader()
        self.writer.writerows(self.rows);self.rows.clear();self.handle.flush()
    def __exit__(self,*exc):
        self.flush()
        if self.handle:self.handle.close()
