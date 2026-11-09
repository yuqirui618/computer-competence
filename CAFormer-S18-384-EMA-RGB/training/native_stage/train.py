"""Train fresh native384 CAFormer on the fixed official split; all paths are explicit."""
from __future__ import annotations
import argparse,os
from pathlib import Path
from types import SimpleNamespace

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','manifest','pretrained-weights','output-dir'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--pretrained-config',type=Path,default=Path(__file__).resolve().parent/'pretrained_config.json')
    p.add_argument('--workers',type=int,choices=(0,1,2,4),default=2)
    g=p.add_mutually_exclusive_group();g.add_argument('--check-only',action='store_true');g.add_argument('--smoke',action='store_true')
    return p.parse_args()

def main(a):
    import common
    common.configure(a)
    import training_core as core
    awake=None
    with core.lock():
        if os.name=='nt':awake=core.ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:core.main(SimpleNamespace(check_only=a.check_only,smoke=a.smoke))
        except KeyboardInterrupt:common.status(state='interrupted',partial_epoch_not_counted=True);raise
        except Exception as exc:common.status(state='failed',error=f'{type(exc).__name__}: {exc}',partial_epoch_not_counted=True);raise
        finally:
            if awake:core.ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main(arguments())
