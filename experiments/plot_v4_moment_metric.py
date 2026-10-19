"""Export an exact illustrative feasible-moment diagram, without empirical claims."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.v4_moment_metric_candidate import metric_projection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    b=3.2;point=torch.tensor([[2.,.6]],dtype=torch.double);base=torch.tensor([b],dtype=torch.double)
    euclidean=metric_projection(point,base,1.,5.,1.,1.)[0].numpy()
    coupled=metric_projection(point,base,1.,5.,1.,.1)[0].numpy()
    proposal=point[0].numpy()
    vertices=np.array([[0.,b-1.],[5.-b,0.],[np.ceil(b)-b,0.],[0.,b-np.floor(b)],[0.,b-1.]])
    table=pd.DataFrame([proposal,euclidean,coupled],columns=['P','N'],index=['proposal','Euclidean','mean-dispersion metric'])
    table['mean_residual']=table.P-table.N;table['dispersion']=table.P+table.N
    table.to_csv(args.output/'moment_metric_geometry.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.4),constrained_layout=True)
    colors=['#384B63','#C37B20','#087F7A'];labels=['Proposal','Euclidean projection','Loss-aligned projection']
    points=[proposal,euclidean,coupled]
    for j,ax in enumerate(axes):
        transform=(lambda x:x) if j==0 else (lambda x:np.stack([x[...,0]-x[...,1],x[...,0]+x[...,1]],-1))
        poly=transform(vertices)
        ax.fill(poly[:,0],poly[:,1],color='#E9EFF4',edgecolor='#66778A',linewidth=1.3)
        ax.plot(*transform(np.stack([proposal,euclidean])).T,linestyle='--',color=colors[1],lw=1.2)
        ax.plot(*transform(np.stack([proposal,coupled])).T,linestyle='--',color=colors[2],lw=1.2)
        for k,p in enumerate(points):
            q=transform(p);ax.scatter(q[0],q[1],s=55,marker=['x','o','s'][k],color=colors[k],label=labels[k],zorder=5)
        ax.grid(alpha=.16);ax.set_axisbelow(True)
    axes[0].set(xlabel='Positive partial moment P',ylabel='Negative partial moment N',title='Same feasible polygon',xlim=(-.1,2.3),ylim=(-.1,2.5))
    axes[1].set(xlabel='Mean coordinate: P - N',ylabel='Dispersion coordinate: P + N',title='Different projection metric')
    axes[0].legend(loc='upper right',frameon=False,fontsize=8)
    fig.suptitle('Moment geometry: support [1, 5], baseline b = 3.2, dispersion weight = 0.1',fontsize=11)
    for ext in ('svg','png'):
        fig.savefig(args.output/('moment_metric_geometry.'+ext),dpi=350)
    plt.close(fig)
    print(table.to_string())


if __name__=='__main__':main()
