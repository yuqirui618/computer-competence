"""Download pinned ImageNet weights to E only; verify repository LFS SHA."""
from pathlib import Path
import urllib.request,json,hashlib,os,time
ROOT=Path(__file__).resolve().parent
assert ROOT.drive.upper()=='E:'
for key in ('TEMP','TMP','TMPDIR','HF_HOME','HF_XET_CACHE','TORCH_HOME','CUDA_CACHE_PATH'):
    path=ROOT/'temp'/key.lower();path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
MODELS={'caformer_s18':('caformer_s18.sail_in1k','apache-2.0'),'densenet121':('densenet121.ra_in1k','apache-2.0'),'hrnet_w18':('hrnet_w18.ms_aug_in1k','mit')}
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Codex-local-model-comparison/1.0'}),timeout=45)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def get_small(url):
    for attempt in range(3):
        try:
            with fetch(url) as f:return f.read()
        except Exception as e:
            print(f'RETRY metadata {attempt+1}: {e}',flush=True)
            if attempt==2:raise
record={'checked_on':'2026-11-05','models':{}}
for key,(architecture,license_name) in MODELS.items():
    target=ROOT/'pretrained'/key;target.mkdir(parents=True,exist_ok=True)
    raw=get_small(f'https://huggingface.co/api/models/timm/{architecture}?blobs=true');info=json.loads(raw)
    assert info.get('private') is False and info.get('gated') is False
    assert info['cardData']['license']==license_name
    revision=info['sha'];entry=next(x for x in info['siblings'] if x['rfilename']=='model.safetensors')
    expected=entry['lfs']['sha256'];size=entry.get('size',entry['lfs'].get('size'))
    (target/'repository_metadata.json').write_bytes(raw)
    for name in ('config.json','README.md'):(target/name).write_bytes(get_small(f'https://huggingface.co/timm/{architecture}/resolve/{revision}/{name}'))
    destination=target/'model.safetensors'
    if not destination.exists() or sha(destination)!=expected:
        part=target/'model.safetensors.part'
        for attempt in range(3):
            try:
                count=0;last=0
                with fetch(f'https://huggingface.co/timm/{architecture}/resolve/{revision}/model.safetensors?download=true') as source,part.open('wb') as out:
                    while chunk:=source.read(1024*1024):
                        out.write(chunk);count+=len(chunk)
                        if count-last>=20*1024*1024:print(f'DOWNLOAD {key} {count}/{size}',flush=True);last=count
                assert part.stat().st_size==size and sha(part)==expected
                part.replace(destination);break
            except Exception as e:
                print(f'RETRY weight {key} {attempt+1}: {e}',flush=True)
                if attempt==2:raise
    record['models'][key]={'architecture':architecture,'directory':key,'sha256':expected,'bytes':size,'repository':f'timm/{architecture}','revision':revision,'license':license_name,'pretraining_data':'ImageNet-1k','source_url':f'https://huggingface.co/timm/{architecture}','config':json.loads((target/'config.json').read_text(encoding='utf-8'))}
    (ROOT/'pretrained_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'WEIGHT_READY {key} {expected}',flush=True)
licenses={
 'LICENSE.apache-2.0.txt':'https://www.apache.org/licenses/LICENSE-2.0.txt',
 'LICENSE.hrnet-MIT.txt':'https://raw.githubusercontent.com/HRNet/HRNet-Image-Classification/master/LICENSE'}
license_records={}
for filename,url in licenses.items():
    body=get_small(url);path=ROOT/'pretrained'/filename;path.write_bytes(body)
    license_records[filename]={'url':url,'sha256':sha(path)}
(ROOT/'pretrained/license_sources.json').write_text(json.dumps(license_records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('ALL_THREE_WEIGHTS_READY',flush=True)
