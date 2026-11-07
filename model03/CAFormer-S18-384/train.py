"""Portable reproduction of A0 BCE+EMA or A2 photo+BCE+EMA, five fresh epochs."""
from __future__ import annotations
import argparse,ast,gc,hashlib,json,math,os,sys,time
from pathlib import Path


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('check','train'))
    p.add_argument('--arm',required=True,choices=('A0_BCE_EMA','A2_photo'))
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--raw-root',required=True,type=Path)
    p.add_argument('--pretrained',required=True,type=Path,help='Pinned ImageNet1K native384 model.safetensors')
    p.add_argument('--output',required=True,type=Path,help='Separate local output; no old experiment files required')
    p.add_argument('--resume',action='store_true',help='Resume this arm at the next committed epoch, at most five total')
    return p.parse_args(argv)


def prepare_output(path,protected_paths=()):
    output=path.expanduser().resolve()
    protected=(Path(__file__).resolve().parent,*(Path(p).expanduser().resolve() for p in protected_paths))
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output must be separate from source, raw data and input-file directories')
    output.mkdir(parents=True,exist_ok=True)
    for key in ('TEMP','TMP','TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','HF_HOME','HF_HUB_CACHE','HF_XET_CACHE','TORCH_HOME','TORCHINDUCTOR_CACHE_DIR'):
        value=output/'cache'/key.lower();value.mkdir(parents=True,exist_ok=True);os.environ[key]=str(value)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    sys.dont_write_bytecode=True
    return output


def source_hashes():
    root=Path(__file__).resolve().parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('train.py','portable_model.py','portable_training.py','data_pipeline.py','augmentation.py')}



def assert_epoch_output_clear(output,epoch):
    for variant in ('raw','ema'):
        if (output/'checkpoints'/f'{variant}_epoch_{epoch:02d}.pth').exists() or (output/'predictions'/f'{variant}_epoch_{epoch:02d}.npz').exists():
            raise ValueError('Uncommitted epoch artifact exists; inspect it before any training retry')

def recipe(args,audit,t):
    return dict(arm=args.arm,architecture='caformer_s18',pretraining='ImageNet1K native384; fresh strict1000 then only fc2 reset10',
        pretrained_sha256=t.SOURCE_SHA,manifest_sha256=t.MANIFEST_SHA,policy=t.POLICIES[args.arm],seed=42,epochs=5,
        real_study_batch=16,accumulation_steps=2,effective_study_batch=32,final_window_studies=17,
        backbone_hidden_lr=5e-5,classifier_lr=5e-4,decay=.05,bias_norm_decay=0.,hidden_drop=.2,outer_drop=0.,drop_path=.1,
        AMP='FP16 forward; stable FP32 complete-image Study mean BCE',EMA='min(.999,(1+successful_updates)/(10+successful_updates))',
        schedule='579 attempts epoch1 linear .1..1; next4 cosine 1..0.1; skipped attempts still advance',
        image_size=384,mean=t.CONFIG['mean'],std=t.CONFIG['std'],loader=t.COMMON,train_priors=audit['train_priors'],source_sha256=source_hashes())


def cpu_checks(args,cfg,records,audit,t):
    import numpy as np
    import torch
    import portable_model as pm
    from PIL import Image
    from data_pipeline import FullFieldTransform
    from augmentation import AugmentedFullFieldTransform
    t.setup();torch.set_num_threads(2)
    checks={};root=Path(__file__).resolve().parent
    model=pm.build_model(args.pretrained,audit['train_priors'],'cpu')
    # Extract just the original function. Its explicit injected source_info uses
    # the supplied file; no historical module or local original path is imported.
    source=(root/'frozen_sources'/'models.py').read_text(encoding='utf-8-sig');tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_model')
    namespace=dict(_libraries=pm._libraries,seed42=pm.seed42,source_info=lambda arch:dict(weights=str(args.pretrained.resolve()),sha256=pm.SOURCE_SHA))
    exec(compile(ast.get_source_segment(source,node),'frozen_models_build_function','exec'),namespace)
    reference=namespace['build_model']('caformer_s18',audit['train_priors'],'cpu')
    state=model.state_dict();ref=reference.state_dict()
    if set(state)!=set(ref) or not all(torch.equal(state[k],ref[k]) for k in state):raise AssertionError('Portable initialization differs from frozen factory')
    checks['all_initial_state_tensors_equal_frozen_factory']=dict(passed=True,tensors=len(state),native_audit=model._native_load_audit)
    opt=pm.optimizer_for(model);coverage=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(coverage)==len(set(coverage)) and set(coverage)=={id(p) for p in model.parameters()}
    checks['optimizer_groups_match']=dict(passed=True,groups=[dict(name=g['name'],lr=g['lr'],weight_decay=g['weight_decay'],tensors=len(g['params'])) for g in opt.param_groups])
    del reference,ref,state,model,opt;gc.collect()
    generator=torch.Generator().manual_seed(42);z=torch.randn(7,10,generator=generator,requires_grad=True);lengths=[2,1,3,1]
    y=torch.randint(0,2,(4,10),generator=generator).float()
    stable=t.losses_for(args.arm,z,lengths,y,None)
    direct=torch.nn.functional.binary_cross_entropy(torch.stack([v.sigmoid().mean(0) for v in z.split(lengths)]),y,reduction='none').mean(1)
    torch.testing.assert_close(stable,direct,atol=1e-6,rtol=1e-6)
    g1,=torch.autograd.grad(stable.sum(),z,retain_graph=True);g2,=torch.autograd.grad(direct.sum(),z)
    torch.testing.assert_close(g1,g2,atol=1e-6,rtol=1e-6)
    features=torch.randn(17,3,generator=generator);yy=torch.randint(0,2,(17,10),generator=generator).float();initial=torch.randn(3,10,generator=generator)
    full=initial.clone().requires_grad_();t.losses_for(args.arm,features@full,[1]*17,yy,None).mean().backward()
    split=initial.clone().requires_grad_()
    for begin,end in ((0,16),(16,17)):(t.losses_for(args.arm,features[begin:end]@split,[1]*(end-begin),yy[begin:end],None).sum()/17).backward()
    torch.testing.assert_close(full.grad,split.grad,atol=1e-6,rtol=1e-6)
    checks['complete_image_Study_BCE_and_last17_gradient']=dict(passed=True,value_error=float((stable-direct).detach().abs().max()),
        gradient_error=float((g1-g2).abs().max()),last17_error=float((full.grad-split.grad).abs().max()))
    toy=torch.nn.Linear(2,1);toy.set_grad_checkpointing=lambda enabled:None;ema=t.ShadowEMA(toy)
    original={k:v.detach().clone() for k,v in toy.state_dict().items()}
    with torch.no_grad():
        for p in toy.parameters():p.add_(.5)
    before={k:v.detach().clone() for k,v in toy.state_dict().items()};ema.update(toy)
    for k in original:
        torch.testing.assert_close(ema.model.state_dict()[k],original[k]*(2/11)+before[k]*(9/11))
        assert torch.equal(toy.state_dict()[k],before[k])
    checks['EMA_recurrence_and_raw_unchanged']=dict(passed=True,first_decay=ema.last_decay,successful_updates=ema.updates)
    for name in ('data_pipeline.py','augmentation.py'):
        assert t.sha(root/name)==t.sha(root/'frozen_sources'/name)
    transformed=0
    for record in records['train'][:2]:
        with Image.open(Path(cfg['data_root'])/record['image_paths'][0]) as image:
            base=FullFieldTransform(384,cfg['mean'],cfg['std'],True)
            none=AugmentedFullFieldTransform(384,cfg['mean'],cfg['std'],True,'none',42)
            t.seed(101);a=base(image);state_a=torch.get_rng_state()
            t.seed(101);b=none(image,key=record['image_paths'][0]);state_b=torch.get_rng_state()
            assert torch.equal(a,b) and torch.equal(state_a,state_b)
            photo=AugmentedFullFieldTransform(384,cfg['mean'],cfg['std'],True,'photo',42);photo.set_epoch(1)
            t.seed(101);c=photo(image,key=record['image_paths'][0]);assert c.shape==(3,384,384) and torch.isfinite(c).all()
        transformed+=1
    checks['unchanged_transforms_none_RNG_and_photo_runtime']=dict(passed=True,official_train_examples=transformed,source_bytes_identical=True)
    # Copied training functions (including inference_mode on predict) retain their body.
    portable_tree=ast.parse((root/'portable_training.py').read_text(encoding='utf-8'))
    for filename,names in {'trainer.py':['lr_factor','log_probabilities','ShadowEMA','training_epoch','losses_for','group_updates'],
                           'training_utils.py':['seed','loader','predict','fit_gates','calibration','metrics']}.items():
        frozen_tree=ast.parse((root/'frozen_sources'/filename).read_text(encoding='utf-8-sig'))
        for name in names:
            pnode=next(n for n in portable_tree.body if getattr(n,'name',None)==name)
            fnode=next(n for n in frozen_tree.body if getattr(n,'name',None)==name)
            assert ast.dump(pnode,include_attributes=False)==ast.dump(fnode,include_attributes=False),name
    checks['training_math_loader_validation_AST_matches_frozen']=dict(passed=True,includes_predict_inference_mode=True)
    from contextlib import redirect_stderr
    from io import StringIO
    from unittest.mock import patch
    parsed=arguments(['train','--arm','A2_photo','--manifest','/a manifest.csv','--raw-root','/raw root',
                      '--pretrained','/weights/model.safetensors','--output','/run output','--resume'])
    assert parsed.arm=='A2_photo' and parsed.resume and parsed.raw_root==Path('/raw root')
    with redirect_stderr(StringIO()):
        try:arguments(['train','--arm','A0_BCE_EMA'])
        except SystemExit as exc:assert exc.code==2
        else:raise AssertionError('Missing explicit paths accepted')
    assert_epoch_output_clear(args.output/'unused_fixture',5)
    with patch('pathlib.Path.exists',return_value=True):
        try:assert_epoch_output_clear(args.output/'unused_fixture',5)
        except ValueError:pass
        else:raise AssertionError('Uncommitted epoch artifact not rejected')
    protected=(args.raw_root,args.manifest.parent,args.pretrained.parent)
    blocked=(root,root/'would_pollute_source',args.raw_root,args.raw_root/'would_pollute_data',args.manifest.parent,args.pretrained.parent)
    for target in blocked:
        with patch('pathlib.Path.mkdir') as mkdir:
            try:prepare_output(target,protected)
            except ValueError:pass
            else:raise AssertionError('Protected output location accepted')
            assert not mkdir.called
    checks['parser_paths_and_pretraining_retry_gate']=dict(passed=True,missing_paths_rejected=True,
        uncommitted_epoch_rejected_before_training=True,protected_output_locations_rejected_before_mkdir=len(blocked))
    assert not torch.cuda.is_initialized()
    receipt=dict(state='passed',mode='CPU check only',checks=checks,coverage=audit['coverage'],train_priors=audit['train_priors'],
        source_sha256=source_hashes(),pretrained_sha256=pm.SOURCE_SHA,training_run=False,GPU_initialized=False,
        no_dependencies_on_old_champion_or_protection_files=True)
    t.save(args.output/'cpu_checks.json',receipt)
    print(json.dumps(dict(state='passed',groups=len(checks),output=str(args.output/'cpu_checks.json')),ensure_ascii=True))


def evaluate(model,variant,epoch,args,cfg,val,t,registration):
    p,extra=t.predict(model,val,cfg,True);measured=t.metrics(val.targets,p)
    cp=args.output/'checkpoints'/f'{variant}_epoch_{epoch:02d}.pth';pp=args.output/'predictions'/f'{variant}_epoch_{epoch:02d}.npz'
    if cp.exists() or pp.exists():raise ValueError('Uncommitted epoch artifact exists; inspect it before retrying')
    t.save_npz(pp,targets=val.targets,probabilities=p,subjects=val.subjects,study_ids=val.study_ids,**extra)
    t.checkpoint(cp,dict(state_dict=t.cpu_state(model),arm=args.arm,variant=variant,epoch=epoch,source_total_epoch=epoch,
        architecture='caformer_s18',pretrained_sha256=t.SOURCE_SHA,registration=registration,
        initialization_kind='fresh ImageNet1k native384',full_backbone_training=True))
    return dict(epoch=epoch,variant=variant,metrics=measured,checkpoint=str(cp),checkpoint_sha256=t.sha(cp),prediction=str(pp),prediction_sha256=t.sha(pp))


def train(args,cfg,records,audit,t):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Training requires CUDA; use check for CPU acceptance')
    t.setup();registered=recipe(args,audit,t)
    registration=dict(recipe=registered,manifest=str(args.manifest.resolve()),raw_root=str(args.raw_root.resolve()),pretrained=str(args.pretrained.resolve()))
    t.freeze(args.output/'registration.json',registration)
    model=t.build_model(args.pretrained,audit['train_priors'],'cuda');initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=t.optimizer_for(model);scaler=torch.amp.GradScaler('cuda',init_scale=1024);ema=t.ShadowEMA(model)
    train_data=t.AugmentedStudyDataset(records['train'],cfg['data_root'],cfg['mean'],cfg['std'],True,policy=t.POLICIES[args.arm],seed=42)
    val=t.RawStudyDataset(records['val'],cfg['data_root'],cfg['mean'],cfg['std'],False)
    history=[];last=args.output/'last_training.pth'
    if args.resume:
        if not last.exists():raise FileNotFoundError(last)
        stored=torch.load(last,map_location='cpu',weights_only=True)
        if stored['registration']!=registration:raise ValueError('Resume recipe/source paths differ')
        history=stored['history']
        if [r['epoch'] for r in history]!=list(range(1,len(history)+1)) or len(history)>5:raise ValueError('Bad committed epoch history')
        for row in history:
            for e in row['evaluations'].values():
                if t.sha(e['checkpoint'])!=e['checkpoint_sha256'] or t.sha(e['prediction'])!=e['prediction_sha256']:raise ValueError('Resume artifact SHA differs')
        model.load_state_dict(stored['state_dict'],strict=True);opt.load_state_dict(stored['optimizer']);scaler.load_state_dict(stored['scaler']);ema.load(stored['ema'])
        del stored
    elif last.exists() or (args.output/'history.json').exists():raise ValueError('Output already contains a run; use --resume')
    profile=dict(real_batch_size=16,accumulation_steps=2)
    for epoch in range(len(history)+1,6):
        if source_hashes()!=registered['source_sha256']:raise ValueError('Portable source changed during training')
        assert_epoch_output_clear(args.output,epoch)
        train_data.set_epoch(epoch)
        row=t.training_epoch(args.arm,model,train_data,opt,scaler,ema,None,epoch,profile,initial,len(history))
        row['evaluations']={variant:evaluate(m,variant,epoch,args,cfg,val,t,registration) for variant,m in (('raw',model),('ema',ema.model))}
        history.append(row)
        t.checkpoint(last,dict(state_dict=t.cpu_state(model),optimizer=opt.state_dict(),scaler=scaler.state_dict(),ema=ema.state(),history=history,registration=registration))
        t.save(args.output/'history.json',history)
        for variant,e in t.best_evaluations(history).items():t.copy_checkpoint(e['checkpoint'],args.output/f'best_{variant}.pth')
        t.status(state='epoch_completed',arm=args.arm,epoch=epoch,raw_AUC=row['evaluations']['raw']['metrics']['macro_auc'],EMA_AUC=row['evaluations']['ema']['metrics']['macro_auc'])
    selected={}
    for variant,e in t.best_evaluations(history).items():
        with __import__('numpy').load(e['prediction'],allow_pickle=False) as d:cal=t.calibration(d['targets'],d['probabilities'],d['subjects'])
        gate=dict(**cal,class_ids=list(range(1,11)),arm=args.arm,architecture='caformer_s18',variant=variant,best_epoch=e['epoch'],
            checkpoint=str(args.output/f'best_{variant}.pth'),checkpoint_sha256=e['checkpoint_sha256'],gate_fit_source='full official validation; patient OOF F1 separate')
        t.save(args.output/f'gates_{variant}.json',gate);selected[variant]=dict(**e,calibration=cal,best_alias=str(args.output/f'best_{variant}.pth'))
    winner=max(selected,key=lambda v:t.key(selected[v]));t.copy_checkpoint(selected[winner]['checkpoint'],args.output/'best.pth')
    gates=t.read(args.output/f'gates_{winner}.json');gates['checkpoint']=str(args.output/'best.pth');t.save(args.output/'gates.json',gates)
    t.save(args.output/'results.json',dict(state='complete',arm=args.arm,completed_new_epochs=5,variants=selected,best_variant=winner,recipe=registered,
        no_test_inference=True,training_epochs_are_fresh=True))
    t.status(state='complete',arm=args.arm,completed_new_epochs=5)


def main():
    args=arguments();args.output=prepare_output(args.output,(args.raw_root,args.manifest.parent,args.pretrained.parent))
    import portable_training as t
    t.configure_output(args.output)
    cfg,records,audit=t.load_inputs(args.manifest,args.raw_root)
    if args.mode=='check':cpu_checks(args,cfg,records,audit,t)
    else:train(args,cfg,records,audit,t)


if __name__=='__main__':main()
