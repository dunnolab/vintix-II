from typing import Any, Dict, List, Optional

import torch
from torch import nn


class TimeFourier(nn.Module):
    """Time Encoder

    Time Encoder with a sinusoidal time embedding

    Args:
        dim: the size of embeddings
        min_freq: minimal frequency value
        max_freq: maximal frequency value
    """
    def __init__(self,
                 dim: int = 64,
                 min_freq: float = 1.0,
                 max_freq: float = 1000.0):
        super().__init__()
        assert dim % 2 == 0
        ks = torch.linspace(0, 1, dim // 2)
        freqs = min_freq * (max_freq / min_freq) ** ks
        self.freqs = nn.Parameter(freqs)

    def forward(self, t):
        a = t * self.freqs
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class AcsField(nn.Module):
    """Action decoder

    Decode action representations into
    continuous actions with MLP

    Args:
        task_metadata: task-specific metadata
        hidden_dim: input model dimension
        emb_activation: hidden activation
        out_activation: output_activation
    """
    def __init__(
            self,
            task_metadata: Dict[str, Any],
            hidden_dim: int,
            emb_activation: nn.Module = nn.LeakyReLU(),
            out_activation: nn.Module = nn.Identity(),
    ):
        super(AcsField, self).__init__()
        self.hidden_dim = hidden_dim
        self.acs_dim = task_metadata['action_dim']

        self.time_emb = TimeFourier(self.hidden_dim)
        self.trans_emb = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.noise_emb = nn.Linear(self.acs_dim, self.hidden_dim)

        # Linear layer case
        self.layers = nn.Sequential(
            emb_activation,
            nn.Linear(3*self.hidden_dim, self.hidden_dim),
            emb_activation,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            emb_activation,
            nn.Linear(self.hidden_dim, self.acs_dim),
            out_activation,
        )

    def forward(self,
                h: torch.Tensor,
                x: torch.Tensor,
                t: torch.Tensor
                ) -> torch.Tensor:
        h = self.trans_emb.forward(h)
        x = self.noise_emb.forward(x)
        t = self.time_emb.forward(t)
        inp = torch.cat([h, x, t], dim=-1)
        out = self.layers(inp)
        return out


class FMDecoder(nn.Module):
    """Flow Matching Decoder for DPT

    Flow Matching Decoder for DPT. Takes the outputs
    of transformer and returns actions.

    Args:
        task2group: task_name-group_name mapping
        task_metadata: task-level metadata for observations,
            actions and rewards
        group_metadata: group-level metadata for observations
            and actions
        hidden_dim: hidden dimension of backbone model
        add_bos: flag of BOS token is added
        emb_activation: activation function for representations
        acs_activation: activation function for actions
    """
    def __init__(
            self,
            task2group: Dict[str, Any],
            task_metadata: Dict[str, Any],
            group_metadata: Dict[str, Any],
            hidden_dim: int,
            unnorm_acs: bool = False,
            add_bos: bool = False,
            emb_activation: nn.Module = nn.LeakyReLU(),
            acs_activation: nn.Module = nn.Identity(),
    ):
        super(FMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.task2group = task2group
        self.task_metadata = task_metadata
        self.group_metadata = group_metadata
        self.unnorm_acs = unnorm_acs
        self.add_bos = add_bos

        self.acs_decoders = nn.ModuleDict(
            {gn: AcsField(
                task_metadata=gm,
                hidden_dim=hidden_dim,
                emb_activation=emb_activation,
                out_activation=acs_activation,
            ) for gn, gm in self.group_metadata.items()}
        )
        self.std_decrease = 1.
        self.steps = 32

    def forward(
        self,
        emb: torch.Tensor,
        metadata: List[Dict[str, Any]],
        gt_actions: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        acs = []
        for inp_num, (traj_emb, mt) in enumerate(zip(emb, metadata)):
            task_name = mt['task_name']
            acs_shape = mt['action_dim']
            if self.add_bos:
                traj_emb = traj_emb[1:]
            noise = torch.randn(traj_emb.shape[0],
                                acs_shape,
                                device=traj_emb.device)
            x0 = self.std_decrease * noise
            if self.training:
                gt_act = gt_actions[inp_num]
                t = torch.rand(traj_emb.shape[0], 1, device=traj_emb.device)

                xt = (1 - t) * x0 + t * gt_act

                field_val = self.acs_decoders[
                    self.task2group[task_name]
                    ].forward(traj_emb, xt, t)
                action = x0 + field_val
            else:
                steps = self.steps
                t_mas = torch.linspace(0, 1, steps+1)
                xt = x0
                for i in range(steps):
                    t = t_mas[i]
                    dt = t_mas[i+1] - t
                    k1 = self.acs_decoders[self.task2group[task_name]].forward(
                        traj_emb,
                        xt,
                        t * torch.ones(traj_emb.shape[0],
                                       1,
                                       device=traj_emb.device)
                    )
                    k2 = self.acs_decoders[self.task2group[task_name]].forward(
                        traj_emb,
                        xt + dt * k1,
                        t_mas[i+1] * torch.ones(traj_emb.shape[0],
                                                1,
                                                device=traj_emb.device)
                    )
                    xt = xt + 0.5 * dt * (k1 + k2)
                action = xt
            acs.append(action)
        return acs
