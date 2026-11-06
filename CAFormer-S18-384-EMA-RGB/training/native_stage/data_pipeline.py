"""Raw official-image Study bags and native384 full-field transforms; no feature/JPEG caches."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

class FullFieldTransform:
    def __init__(self,size,mean,std,is_train=False):
        if size!=384:raise ValueError('Native384 input required')
        self.size=size;self.is_train=is_train
        self.affine=transforms.RandomAffine(degrees=5,translate=(.02,.02),scale=(.95,1.05),interpolation=transforms.InterpolationMode.BICUBIC,fill=0) if is_train else None
        self.tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize(tuple(mean),tuple(std))])
    def __call__(self,image):
        if image.mode in ('I','F') or image.mode.startswith('I;16'):raise ValueError('High-bit-depth image requires explicit normalization; refusing silent RGB truncation')
        # RandomAffine receives PIL, where BICUBIC is supported; tensor affine is not used.
        image=ImageOps.pad(image.convert('RGB'),(self.size,self.size),method=Image.Resampling.BICUBIC,color=0)
        if self.affine is not None:image=self.affine(image)
        return self.tensor(image)

class RawStudyDataset(Dataset):
    def __init__(self,records,data_root,mean,std,is_train=False):
        self.records=records;self.root=Path(data_root);self.transform=FullFieldTransform(384,mean,std,is_train)
        self.targets=np.asarray([r['target'] for r in records],dtype=np.float32)
        self.study_ids=np.asarray([r['study_id'] for r in records],dtype=str)
        self.subjects=np.asarray([r['subject_id'] for r in records],dtype=str)
        self.lengths=np.asarray([len(r['image_paths']) for r in records],dtype=np.int64)
        self.images=int(self.lengths.sum())
        if len(set(self.study_ids))!=len(records) or not (self.lengths>0).all():raise ValueError('Invalid Study records')
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        images=[]
        for relative in self.records[index]['image_paths']:
            with Image.open(self.root/relative) as image:images.append(self.transform(image))
        return torch.stack(images),torch.from_numpy(self.targets[index]),index

class RawImageDataset(Dataset):
    def __init__(self,records,data_root,mean,std):
        self.root=Path(data_root);self.transform=FullFieldTransform(384,mean,std,False)
        entries=[]
        for index,record in enumerate(records):
            for position,path in zip(record['image_positions'],record['image_paths']):entries.append((position,path,index))
        entries.sort(key=lambda x:x[0])
        self.paths=[x[1] for x in entries];self.study_indices=[x[2] for x in entries]
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        with Image.open(self.root/self.paths[index]) as image:value=self.transform(image)
        return value,index

def collate_studies(batch):
    images=torch.cat([x[0] for x in batch],0);lengths=[len(x[0]) for x in batch]
    return images,lengths,torch.stack([x[1] for x in batch]),torch.tensor([x[2] for x in batch],dtype=torch.int64)

def worker_init(worker_id):
    value=torch.initial_seed()%2**32;random.seed(value);np.random.seed(value);torch.set_num_threads(1)
