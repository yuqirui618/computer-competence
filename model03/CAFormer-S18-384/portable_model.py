"""Portable native384 S18 factory: explicit local ImageNet path, no downloads."""
from __future__ import annotations
import hashlib,random
from pathlib import Path
SOURCE_SHA='629149fd6170274f8f19d1fbbb9b155b13933ff3d2f9e991ac0339d993426806'
SOURCE_REPOSITORY='timm/caformer_s18.sail_in1k_384'
SOURCE_REVISION='c46cc16d67a7d68806b7554b93e16b13506b7035'


def sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def _libraries():
    import numpy as np
    import torch,timm
    from safetensors.torch import load_file
    return np,torch,timm,load_file

def seed42():
    np,torch,_,_=_libraries()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)

def build_model(pretrained,priors,device='cpu'):
    architecture='caformer_s18'
    path=Path(pretrained).expanduser().resolve()
    if sha256(path)!=SOURCE_SHA:raise ValueError('Expected pinned ImageNet1K native384 safetensors')
    meta=dict(weights=str(path),sha256=SOURCE_SHA)
    np,torch,timm,load_file=_libraries()
    values=torch.as_tensor(priors,dtype=torch.float32,device='cpu')
    if values.shape!=(10,) or not torch.isfinite(values).all() or not ((values>0)&(values<1)).all():
        raise ValueError('Ten finite train-Study priors strictly between zero and one required')
    seed42()
    model=timm.create_model(architecture,pretrained=False,num_classes=1000,
        drop_rate=0.,drop_path_rate=.1,proj_drop_rate=0.)
    source=load_file(meta['weights'],device='cpu')
    model.load_state_dict(source,strict=True)
    initial=model.state_dict()
    if any(not torch.equal(value,initial[key]) for key,value in source.items()):
        raise ValueError('Strict1000 native source load was not tensor-exact')
    if model.head.fc.fc1.out_features!=2048 or model.head.fc.fc2.in_features!=2048 or model.head.fc.fc2.out_features!=1000:
        raise ValueError('Full original 2048 hidden MLP/classifier differs')
    model.head.fc.head_drop.p=.2
    if getattr(model.head.drop,'p',0.)!=0.:raise ValueError('Unexpected outer-head dropout')
    # Same RNG sequence as the original native S18 build_model: constructor then
    # strict source load (no RNG), then Linear default reset and trunc_normal_.
    layer=torch.nn.Linear(2048,10)
    torch.nn.init.trunc_normal_(layer.weight,std=.02)
    with torch.no_grad():layer.bias.copy_(torch.logit(torch.tensor(priors,dtype=torch.float32)))
    model.head.fc.fc2=layer;model.num_classes=10
    model.requires_grad_(True);model.set_grad_checkpointing(True)
    if any(isinstance(module,torch.nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        raise ValueError('BatchNorm requires separate real-batch review')
    if not all(parameter.requires_grad for parameter in model.parameters()):raise ValueError('Unexpected frozen parameters')
    unchanged=sum(torch.equal(value,model.state_dict()[key]) for key,value in source.items() if not key.startswith('head.fc.fc2.'))
    expected=sum(not key.startswith('head.fc.fc2.') for key in source)
    if unchanged!=expected:raise ValueError('An original body/hidden-head tensor was changed')
    model._native_load_audit=dict(architecture=architecture,source_sha256=meta['sha256'],
        strict_1000_tensor_count=len(source),strict_1000_all_tensors_equal=True,
        preserved_non_fc2_tensors=unchanged,expected_non_fc2_tensors=expected,
        only_final_fc2_replaced=True,initialization_seed=42)
    if str(device)=='cuda':model.to(device='cuda',memory_format=torch.channels_last)
    elif str(device)=='cpu':model.to(device='cpu')
    else:raise ValueError("Use 'cpu' or 'cuda'; resource checks are owned by the runner")
    return model

def optimizer_for(model):
    _,torch,_,_=_libraries();groups={}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:raise ValueError('Full backbone/hidden head must be trainable')
        classifier=name.startswith('head.fc.fc2.')
        decay=parameter.ndim>1 and not name.endswith('.bias')
        key=('classifier' if classifier else 'backbone')+('_decay' if decay else '_no_decay')
        rate=5e-4 if classifier else 5e-5
        if key not in groups:groups[key]=dict(params=[],lr=rate,base_lr=rate,weight_decay=.05 if decay else 0.,name=key)
        groups[key]['params'].append(parameter)
    assigned=[id(parameter) for group in groups.values() for parameter in group['params']]
    if len(assigned)!=len(set(assigned)) or set(assigned)!={id(p) for p in model.parameters()}:
        raise ValueError('Optimizer must cover every parameter exactly once')
    return torch.optim.AdamW(list(groups.values()))
