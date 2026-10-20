"""Draw a new figure from exported CSV/Excel without retraining any model."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path);p.add_argument('--sheet',default='metrics_long')
    p.add_argument('--x',required=True);p.add_argument('--y',required=True);p.add_argument('--group')
    p.add_argument('--kind',choices=['line','scatter','barh','box','heatmap'],default='barh')
    p.add_argument('--value');p.add_argument('--aggregate',choices=['mean','median']);p.add_argument('--where',action='append',default=[])
    p.add_argument('--xlabel');p.add_argument('--ylabel');p.add_argument('--title',default='')
    p.add_argument('--log-x',action='store_true');p.add_argument('--width',type=float,default=6.6);p.add_argument('--height',type=float,default=4.)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    if a.input.suffix.lower()=='.xlsx':
        book=pd.ExcelFile(a.input);sheet=a.sheet
        if sheet not in book.sheet_names and 'INDEX' in book.sheet_names:
            index=pd.read_excel(book,sheet_name='INDEX')
            found=index[index.source.map(lambda s:Path(str(s)).stem)==sheet]
            if len(found)!=1:raise ValueError('Choose an exact sheet from workbook INDEX')
            sheets=str(found.iloc[0].sheets).split(', ')
            frame=pd.concat([pd.read_excel(book,sheet_name=s) for s in sheets],ignore_index=True)
        else:frame=pd.read_excel(book,sheet_name=sheet)
    else:frame=pd.read_csv(a.input)
    for predicate in a.where:
        name,sep,value=predicate.partition('=')
        if not sep or name not in frame:raise ValueError('Filter must be column=value')
        frame=frame[frame[name].astype(str)==value]
    columns=[a.x,a.y]+([a.group] if a.group else [])+([a.value] if a.value else [])
    if any(c not in frame for c in columns):raise ValueError('Unknown requested column')
    data=frame[list(dict.fromkeys(columns))].dropna().copy()
    if data.empty:raise ValueError('No data after filters; no figure was fabricated')
    if a.aggregate and a.kind not in ('heatmap','box'):
        keys=list(dict.fromkeys([a.x]+([a.group] if a.group else [])))
        data=data.groupby(keys,as_index=False,dropna=False)[a.y].agg(a.aggregate)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from v5.publication import STYLE
    from v5.reporting import COLORS
    from v5.figures import _save
    with plt.rc_context(STYLE):
        fig,ax=plt.subplots(figsize=(a.width,a.height),layout='constrained')
        pieces=data.groupby(a.group,sort=False) if a.group else [('data',data)]
        if a.kind=='heatmap':
            if not a.value:raise ValueError('Heatmap requires --value')
            matrix=data.pivot_table(index=a.y,columns=a.x,values=a.value,aggfunc=a.aggregate or 'mean')
            image=ax.imshow(matrix.to_numpy(),aspect='auto',cmap='viridis')
            ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=45,ha='right');ax.set_yticks(range(len(matrix.index)),matrix.index)
            fig.colorbar(image,ax=ax,label=a.value)
        elif a.kind=='box':
            labels=list(data[a.x].unique());ax.boxplot([data.loc[data[a.x]==label,a.y] for label in labels],tick_labels=labels,orientation='horizontal',showfliers=False)
        else:
            for group,line in pieces:
                color=COLORS.get(str(group),'#0072B2');label=str(group) if a.group else None
                if a.kind=='line':
                    line=line.sort_values(a.x)
                    ax.plot(line[a.x],line[a.y],marker='o',label=label,color=color)
                elif a.kind=='scatter':ax.scatter(line[a.x],line[a.y],s=24,label=label,color=color,alpha=.8)
                else:
                    line=line.sort_values(a.y);ax.barh(line[a.x].astype(str),line[a.y],color=color,alpha=.85,label=label,height=.68)
        ax.set(title=a.title,xlabel=a.xlabel or (a.y if a.kind in ('barh','box') else a.x),ylabel=a.ylabel or (a.x if a.kind in ('barh','box') else a.y))
        ax.grid(axis='x' if a.kind in ('barh','box') else 'both');ax.set_axisbelow(True)
        if a.log_x:ax.set_xscale('log')
        if a.group:ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=7)
        _save(fig,a.output.parent,a.output.stem)
    data.to_csv(a.output.with_suffix('.csv'),index=False,float_format='%.17g')
    a.output.with_suffix('.json').write_text(json.dumps(dict(source=str(a.input),source_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},plotted_rows=len(data),training_performed=False),indent=2))


if __name__=='__main__':main()
