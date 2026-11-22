"""Deterministic distributed sampling, DDP launch state and atomic checkpoints."""
from __future__ import annotations
import os, json, hashlib, itertools
from pathlib import Path
import numpy as np


def init_distributed():
    import torch
    import torch.distributed as dist
    rank, world, local = (int(os.environ.get(k, v)) for k, v in [('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)])
    device = f'cuda:{local}' if torch.cuda.is_available() else 'cpu'
    if device != 'cpu': torch.cuda.set_device(local)
    if world > 1: dist.init_process_group('nccl' if device != 'cpu' else 'gloo')
    return rank, world, local, device


def rank_indices(dataset, steps, global_batch, rank, world, start_step=0):
    """One deterministic global stream, partitioned without duplicate rank assignment.

    Replacement within the scientific site-balanced sampler remains intentional.
    Changing world size preserves the global draw order, but exact RNG resume
    requires the original world size and micro-batch configuration.
    """
    if global_batch % world: raise ValueError('global_batch must divide world_size')
    stream=dataset.site_balanced_stream(steps*global_batch, start=start_step*global_batch)
    return np.fromiter(itertools.islice(stream,rank,None,world),dtype=np.int64)


def atomic_torch_save(payload,path):
    import torch
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    torch.save(payload,tmp); os.replace(tmp,path)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


class TensorDataset:
    """Top-level, spawn-picklable dataset. Includes untouched scoring targets."""
    def __init__(self, dataset, indices=None):
        self.dataset=dataset; self.indices=indices
    def __len__(self): return len(self.dataset) if self.indices is None else len(self.indices)
    def __getitem__(self,index):
        import torch
        idx=int(index if self.indices is None else self.indices[index])
        item=self.dataset.build_item(idx)
        for view in item['views']:
            for key in ('img','ray_directions'): view[key]=torch.from_numpy(view[key])
        for key in ('target_depth_z','target_valid'): item[key]=torch.from_numpy(item[key])
        return item


def optimizer_config_identity(config,world):
    fields=('max_steps','global_batch_size','batch_size','learning_rate','weight_decay',
            'warmup_steps','huber_delta','grad_clip_norm','amp_dtype','trainable',
            'seed','height','width','eval_every','gradient_checkpointing','deterministic')
    return {**{k:config.get(k) for k in fields},'world_size':world}


def training_source_identity(root):
    """Hash training/data code, independent of output files and repository location."""
    root = Path(root)
    files = [root / 'scripts/train_distributed.py']
    for directory in ('train', 'data', 'score'):
        files.extend((root / 'src/egodepth_eval' / directory).rglob('*.py'))
    files.append(root / 'src/egodepth_eval/infer/mapanything_backend.py')
    return canonical_digest([(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest())
                             for p in sorted(set(files))])
