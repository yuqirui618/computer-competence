"""Portable BCE+EMA continuation of a complete native384 epoch5 checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','manifest','initial-checkpoint','initial-predictions','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--initial-sha256',default='e4396db33d9fa25751205fc4d848766ff894ec4d104d884b77847e4afd01b250')
    p.add_argument('--initial-predictions-sha256',default='00e56139a419cfd76ae4985896796325daffc3415050dec411acbfdf77ccfe10')
    p.add_argument('--workers',type=int,choices=(0,1,2,4),default=2)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only',action='store_true')
    group.add_argument('--smoke',action='store_true')
    group.add_argument('--run',action='store_true')
    return p.parse_args()

def main(args):
    import ema_common
    ema_common.configure(args)
    import ema_core
    with ema_core.gpu_lock():
        ema_core.main(args)
if __name__=='__main__':main(arguments())
