"""Console progress for candidate training: exact counts, method, K, seed, init and validation ticks.

Instrumentation wraps functions during a study, never alters the scientific losses or optimizers.
"""
from __future__ import annotations
import csv, os, pathlib, threading, time
from contextlib import contextmanager

class CandidateProgress:
    def __init__(self,total,stage,log_path,epoch_train_rmse=False,heartbeat=25):
        self.total=max(1,int(total));self.stage=str(stage);self.done=0
        self.start=time.monotonic();self.active=None;self.current_split_seed=None;self.epoch_train_rmse=epoch_train_rmse
        self.log_path=pathlib.Path(log_path);self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.heartbeat=float(heartbeat);self._stop=threading.Event();self._thread=None
        self._originals={};self._lock=threading.Lock()
        self.n_epoch_records=0
    def emit(self,label=''):
        with self._lock:
            elapsed=time.monotonic()-self.start;frac=self.done/self.total
            eta=(elapsed/max(self.done,1))*(self.total-self.done) if self.done else float('nan')
            a=self.active or {}
            desc=f"{a.get('method','')} split={a.get('split_seed','?')} K={a.get('k','?')} seed={a.get('seed','?')} init={a.get('init','?')}"
            elapsed_label=f'{elapsed/60:.1f}m';eta_label=f'{eta/60:.1f}m' if self.done else 'calculating'
            print(f'[{self.stage}] {self.done}/{self.total} {frac*100:6.2f}% | elapsed={elapsed_label} ETA={eta_label} | {label} {desc}',flush=True)
    def _pulse(self):
        while not self._stop.wait(self.heartbeat):
            self.emit('HEARTBEAT')
    def on_eval(self, source, original_fn, args, kwargs, result):
        a=self.active
        if a is None or not a.get('training') or len(args)<3 or args[0] is not a.get('model') or args[2]!='val':
            return result
        a['val_ticks']+=1
        interval=max(1,int(a['eval_every']))
        epoch=1 if a['val_ticks']==1 else (a['val_ticks']-1)*interval
        val=result[0] if isinstance(result,tuple) else result
        try: vr=float(val['RMSE'])
        except (KeyError,ValueError,TypeError):return result
        tr=''
        if self.epoch_train_rmse:
            # Call the unpatched original evaluation directly to avoid recursion.
            # core.evaluate_model and pub.evaluate_publication_model have distinct APIs.
            try:
                train=original_fn(args[0],args[1],'train')
                tr=float(train['RMSE'] if isinstance(train,dict) else train[0]['RMSE'])
            except Exception as exc:
                tr=f'ERROR:{type(exc).__name__}'
        with self.log_path.open('a',newline='',encoding='utf8') as f:
            w=csv.writer(f)
            if f.tell()==0:w.writerow(['stage','method','split_seed','K','model_seed','init','epoch','train_RMSE','val_RMSE','source'])
            w.writerow([self.stage,a['method'],a.get('split_seed'),a['k'],a['seed'],a['init'],epoch,tr,vr,source])
        self.n_epoch_records+=1
        if a['val_ticks']==1 or a['val_ticks']%5==0:
            self.emit(f'EPOCH {epoch} valRMSE={vr:.5f}')
        return result

    @contextmanager
    def instrument(self,pub,core):
        old_inst=pub.instantiate_model_v8;old_train=pub.train_model_v8
        old_pub_eval=pub.evaluate_publication_model;old_core_eval=core.evaluate_model
        old_make_split=pub.make_core_split
        def make_split(raw,spec,dc):
            self.current_split_seed=int(dc.split_seed)
            return old_make_split(raw,spec,dc)
        def inst(method,data,cfg,k,model_seed,device,init_mode=None):
            model,mcfg=old_inst(method,data,cfg,k,model_seed,device,init_mode)
            self.active={'model':model,'method':method,'split_seed':self.current_split_seed,'k':k,'seed':model_seed,
                'init':init_mode or 'NA','training':False,'val_ticks':0,
                'eval_every':getattr(mcfg,'eval_every',getattr(cfg.core_config.paper,'eval_every',5))}
            self.emit('START')
            return model,mcfg
        def train(method,model,data,mcfg,pcfg):
            if self.active is not None:self.active['training']=True
            try:return old_train(method,model,data,mcfg,pcfg)
            finally:
                if self.active is not None:self.active['training']=False
                self.done+=1;self.emit('FINISHED')
        def pub_eval(*a,**kw):
            out=old_pub_eval(*a,**kw)
            return self.on_eval('sparse-or-baseline',old_pub_eval,a,kw,out)
        def core_eval(*a,**kw):
            out=old_core_eval(*a,**kw)
            return self.on_eval('core-dense',old_core_eval,a,kw,out)
        pub.make_core_split=make_split
        pub.instantiate_model_v8=inst;pub.train_model_v8=train
        pub.evaluate_publication_model=pub_eval;core.evaluate_model=core_eval
        self._thread=threading.Thread(target=self._pulse,daemon=True);self._thread.start()
        try:yield self
        finally:
            self._stop.set()
            pub.make_core_split=old_make_split
            pub.instantiate_model_v8=old_inst;pub.train_model_v8=old_train
            pub.evaluate_publication_model=old_pub_eval;core.evaluate_model=old_core_eval
            if self._thread:self._thread.join(timeout=1)
            self.emit('STAGE END')
