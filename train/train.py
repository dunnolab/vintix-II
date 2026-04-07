import datetime
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import pyrallis
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from train.data.torch_dataloaders import MultiTaskMapDataset
from train.schedule import cosine_annealing_with_warmup
from train.train_utils import (compute_stats,
                               configure_optimizers,
                               initialize_model,
                               train_loop,
                               set_seed)


@dataclass
class TrainConfig:
    # dataloader config
    data_dir: str = "/home/jovyan/foundation_datasets_new"
    context_len: int = 8192
    trajectory_sparsity: int = 64
    ignore_episode_sparsity: Optional[bool] = False
    preload: bool = True
    ic_treshold: float = 0.3
    last_frac: Optional[float] = None
    exp_treshold: float = 0.75

    # model config
    action_emb_dim: int = 255
    observation_emb_dim: int = 255
    reward_emb_dim: int = 2
    hidden_dim: int = 512  # a_dim+o_dim+r_dim
    transformer_depth: int = 16
    transformer_heads: int = 16
    attn_dropout: float = 0
    residual_dropout: float = 0
    normalize_qk: bool = False
    bias: bool = True
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    norm_class: str = "LayerNorm"
    mlp_class: str = "GptNeoxMLP"
    intermediate_size: int = 512
    inner_ep_pos_enc: bool = False
    norm_acs: bool = False
    norm_obs: bool = False
    add_bos: bool = False
    trans_prior: bool = False

    # optimizer config
    optimizer: str = "Adam"
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1e-2
    precision: str = "bf16"
    clip_grad: Optional[float] = 1.0
    grad_accum_steps: int = 1
    warmup_ratio: float = 0.001

    # training config
    local_rank: int = 0
    epochs: int = 200
    batch_size: int = 12
    eval_every: int = 5
    val_max_trans: int = 80000
    smooth_win: int = 8
    save_every: int = 5
    save_dir: str = "/home/jovyan/found_model_checkpoints/"
    stats_path: str = ""
    load_ckpt: Optional[str] = None
    start_epoch: int = 11
    seed: int = 5
    permute: bool = False

    # Dataset config
    dataset_config_paths: List[str] = None

    # wandb config
    project: str = "foundation_rl"
    group: str = "default"
    name: str = "train_ad_d16h16_hs383_wd2_no"

    def __post_init__(self):
        self.dataset_config = {}
        for dcp in self.dataset_config_paths:
            dc = OmegaConf.load(dcp)
            self.dataset_config = {**self.dataset_config, **dc}
        self.dataset_names = {v.path: v.group
                              for k, v in self.dataset_config.items()
                              if v.type == "default"}
        self.reward_scales = [
            v.reward_scale
            for v in self.dataset_config.values()
            if v.type == "default"]
        self.episode_sparsity = [
            v.episode_sparsity if not self.ignore_episode_sparsity else 1
            for v in self.dataset_config.values()
            if v.type == "default"]
        self.tn2en = {
            os.path.basename(v.path): k
            for k, v in self.dataset_config.items()
            if v.type == "default"}


HALF_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed)
    dist.init_process_group(backend='nccl',
                            timeout=datetime.timedelta(minutes=90))
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    config.local_rank = device_id
    num_gpus = int(os.getenv("WORLD_SIZE", "1"))

    print(f"NCCL Init on rank {rank}")

    if config.local_rank == 0:
        config.save_dir = os.path.join(config.save_dir, config.name)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        _ = wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            config=asdict(config),
            save_code=True,
        )
        print("COMPUTING STATS")
        stats = compute_stats(config.data_dir, config.dataset_names)
        with open(config.stats_path, 'w') as f:
            json.dump(stats, f)

    print("PREPARING DATASET")
    dataset = MultiTaskMapDataset(
        data_dir=config.data_dir,
        datasets_info=config.dataset_names,
        trajectory_len=config.context_len,
        trajectory_sparsity=config.trajectory_sparsity,
        ep_sparsity=config.episode_sparsity,
        last_frac=config.last_frac,
        preload=config.preload,
        add_demonstrator_actions=True,
        permute=config.permute,
        randomized=True
    )
    print("DATASET LEN: ", len(dataset))
    sampler = DistributedSampler(
        dataset,
        shuffle=True,
        drop_last=True,
        num_replicas=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(config.local_rank),
        seed=0
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: x,
        sampler=sampler,
        drop_last=True,
        num_workers=12
    )

    print("INITIALIZING MODEL")
    scaler = torch.cuda.amp.GradScaler()
    torch_dtype = HALF_DTYPES[config.precision]
    print("BEFORE BARRIER")
    dist.barrier()
    print("AFTER BARRIER")
    device = torch.device("cuda", config.local_rank)
    model, _ = initialize_model(config,
                                dataset.metadata,
                                device)
    model = DDP(model,
                device_ids=[config.local_rank],
                find_unused_parameters=True)
    optimizer = configure_optimizers(config, model)
    if config.load_ckpt:
        opt_dct = torch.load(os.path.join(config.load_ckpt, "state.pth"),
                             map_location='cpu')
        optimizer.load_state_dict(opt_dct["optimizer_state"])

    if config.local_rank == 0:
        wandb.watch(model, log_freq=500)
    total_updates = len(dataloader) * config.epochs
    print("TOTAL UPDATES: ", total_updates)
    scheduler = cosine_annealing_with_warmup(
        optimizer=optimizer,
        warmup_steps=int(total_updates * config.warmup_ratio),
        total_steps=total_updates,
    )

    print("MODEL INIT COMPLETE")
    print(f"PARAMETERS: {sum(p.numel() for p in model.parameters())}")
    print(f"Device {config.local_rank}")
    dist.barrier()

    if config.local_rank == 0:
        folder_name = "0".zfill(4) + "_epoch"
        model.module.save_model(
            os.path.join(config.save_dir, folder_name)
        )
        torch.save(
            {"optimizer_state": optimizer.state_dict(),
             "scheduler_state": scheduler.state_dict()},
            os.path.join(config.save_dir, folder_name, "state.pth")
        )

    for ep in range(config.start_epoch, config.epochs):
        sampler.set_epoch(ep)
        train_loop(config,
                   ep,
                   num_gpus,
                   model,
                   optimizer,
                   scheduler,
                   scaler,
                   dataloader,
                   device,
                   torch_dtype,
                   distributed=True)
        if ep % config.save_every == 0 and config.local_rank == 0:
            folder_name = f"{ep}".zfill(4) + "_epoch"
            model.module.save_model(
                os.path.join(config.save_dir, folder_name)
            )
            torch.save(
                {"optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict()},
                os.path.join(config.save_dir, folder_name, "state.pth")
            )
        dist.barrier()


if __name__ == '__main__':
    train()
