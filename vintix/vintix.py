import json
import os
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from vintix.model.dpt_individual_task_encoder import DPTIndividualTaskEncoder

from vintix.model.dpt_individual_task_head import FMDecoder
from vintix.model.model import Transformer


class Vintix2(nn.Module):

    """DPT Action Model"""
    def __init__(self):
        super(Vintix2, self).__init__()
        self.metadata = {}
        self.conf = {}

    def init_model(self,
                   config: Dict[str, Any],
                   metadata: Dict[str, Any]):
        """Initialize model

        Initialization of the DPT model

        Args:
            config: model config
            metadata: metadata of tasks
        """
        self.metadata = copy(metadata)
        self.conf = config
        task2group = {}
        group_metadata = {}
        for tn, info in self.metadata.items():
            task2group[tn] = info["group_name"]
            if info["group_name"] not in group_metadata:
                group_metadata[info["group_name"]] = info

        if "add_bos" not in self.conf:
            self.conf["add_bos"] = False

        context_len = self.conf['context_len']

        if self.conf["add_bos"]:
            context_len += 1

        self.encoder = DPTIndividualTaskEncoder(
            task2group=task2group,
            task_metadata=self.metadata,
            group_metadata=group_metadata,
            action_emb_dim=self.conf['action_emb_dim'],
            observation_emb_dim=self.conf['observation_emb_dim'],
            reward_emb_dim=self.conf['reward_emb_dim'],
            out_seq_len=context_len,
            inner_ep_pos_enc=self.conf["inner_ep_pos_enc"],
            norm_acs=self.conf["norm_acs"],
            norm_obs=self.conf["norm_obs"],
            emb_activation=nn.LeakyReLU(),
            add_bos=self.conf["add_bos"]
        )
        self.transformer = Transformer(
            hidden_dim=self.conf['hidden_dim'],
            num_layers=self.conf['transformer_depth'],
            num_heads=self.conf['transformer_heads'],
            seq_len=context_len,
            attention_dropout=self.conf['attn_dropout'],
            residual_dropout=self.conf['residual_dropout'],
            normalize_qk=self.conf['normalize_qk'],
            bias=self.conf['bias'],
            parallel_residual=self.conf['parallel_residual'],
            shared_attention_norm=self.conf['shared_attention_norm'],
            norm_class=self.conf['norm_class'],
            mlp_class=self.conf['mlp_class'],
            intermediate_size=self.conf['intermediate_size'],
        )
        self.head = FMDecoder(
            task2group=task2group,
            task_metadata=self.metadata,
            group_metadata=group_metadata,
            hidden_dim=self.conf['hidden_dim'],
            unnorm_acs=self.conf["norm_acs"],
            add_bos=self.conf["add_bos"],
            emb_activation=nn.LeakyReLU(),
            acs_activation=nn.Identity(),
        )

    def forward(self,
                batch: Dict[str, Any]) -> List[torch.Tensor]:
        embs = self.encoder.forward(batch)
        trans_out = self.transformer(
            embs['input_tokens'])
        pred_acs = self.head.forward(
            emb=trans_out,
            metadata=embs['metadata'],
            gt_actions=[t['demonstrator_action'] for t in batch])
        return pred_acs, embs['metadata']

    def add_task(self,
                 task_name: str,
                 group_name: str,
                 stats: Dict[str, List[float]],
                 rew_scale: float = 1) -> None:
        """Add information about task into model

        Args:
            task_name: name of the new task
            group_name: name of the new task's group
            stats: statistics of the new task (mean/std of
                observations and actions)
            rew_scale: reward scale of the new task
        """
        if task_name not in self.encoder.task2group:
            self.encoder.task2group[task_name] = group_name
        if task_name not in self.encoder.task_metadata:
            self.encoder.task_metadata[task_name] = {
                "reward_scale": rew_scale
            }
            for key in stats[task_name].keys():
                stat = stats[task_name][key]
                self.encoder.task_metadata[task_name][key] = stat
        if task_name not in self.head.task2group:
            self.head.task2group[task_name] = group_name
        if task_name not in self.head.task_metadata:
            for key in stats[task_name].keys():
                stat = stats[task_name][key]
                self.head.task_metadata[task_name][key] = stat

    @torch.no_grad()
    def get_action(self,
                   model_input: List[Dict[str, Any]]
                   ) -> List[np.ndarray]:
        """Predict action

        Predict next actions for current input

        Args:
            model_input: model input dicts. Dicts' signature
                is following:
                {
                    "observation": torch.Tensor - observations sequence
                    "prev_action": torch.Tensor - previous actions sequence
                    "prev_reward": torch.Tensor - previous rewards sequence
                    "step_num":  torch.Tensor - within episode step number
                        sequence
                    "task_name": str - name of task
                }
                Such dicts can be created in `create_model_input` method.
            cache: kv-cache

        Returns:
            List[np.ndarray]: list of predicted actions
        """

        embs = self.encoder.forward(model_input)
        trans_out = self.transformer(embs['input_tokens'])

        preds = self.head.forward(
            emb=trans_out,
            metadata=embs['metadata'])
        model_action = [pred[-1].detach().cpu().type(torch.float32).numpy()
                        for pred in preds]
        return model_action

    def save_model(self, path: str) -> None:
        """Save model to folder"""
        os.makedirs(path, exist_ok=True)

        self.conf["model_type"] = "FMDPT"

        model_path = os.path.join(path, 'model.pth')
        metadata_path = os.path.join(path, 'metadata.json')
        config_path = os.path.join(path, 'config.json')
        torch.save(
            self.state_dict(),
            model_path
        )
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        with open(config_path, 'w') as f:
            json.dump(self.conf, f)

    def load_model(self, path: str) -> None:
        """Load model from folder"""
        model_path = os.path.join(path, 'model.pth')
        metadata_path = os.path.join(path, 'metadata.json')
        config_path = os.path.join(path, 'config.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.init_model(config, metadata)
        state_dict_old = torch.load(model_path,
                                    map_location='cpu',
                                    weights_only=True)
        self.load_state_dict(state_dict_old)

    def create_model_input(self,
                           observations: torch.Tensor,
                           actions: torch.Tensor,
                           rewards: torch.Tensor,
                           step_nums: torch.Tensor,
                           task_name: str) -> List[Dict[str,
                                                        torch.Tensor]]:
        """Create appropriate input for `get_action` method

        Args:
            observations: observations sequence
            actions: previous actions sequence
            rewards: previous rewards sequence
            step_nums: within episode step number sequence
            task_name: the name of task
        """

        inp = {'observation': observations[1:],
               'prev_action': actions,
               'prev_reward': rewards.unsqueeze(1),
               'step_num': step_nums[1:],
               'task_name': task_name,
               'perm': np.arange(observations.shape[0] - 2)}
        return [inp]

    def reset_context(self,
                      task_name: str,
                      torch_dtype: torch.dtype = torch.float16,
                      prompt: Optional[Tuple[List[np.ndarray],
                                             List[np.ndarray],
                                             List[float],
                                             List[int]]] = None
                      ) -> None:
        """Reset model context

        Reset stored observations, actions, rewards and step numbers;
        Set new task parameters for prediction, set new prompt if passed

        Args:
            task_name: name of task (required fore defining a group)
            torch_dtype: which torch dtype should be used for computing
                (torch.float16/torch.bfloat16 is possible)
            prompt: tuple of observation sequence, action sequence,
                reward sequence and step number sequence
        """

        assert task_name in self.encoder.task2group
        assert torch_dtype in [torch.float16, torch.bfloat16]

        group = self.encoder.task2group[task_name]
        group_info = self.encoder.group_metadata[group]
        self._acs_dim = group_info['action_dim']
        self._obs_dim = group_info['observation_shape']['proprio'][0]
        self._seq_len = self.conf['context_len'] - 2
        self._np_dtype = np.float32
        self._torch_dtype = torch_dtype
        self._task_name = task_name
        self._device = next(self.parameters()).device

        self._observations = torch.from_numpy(
            np.zeros((1, self._obs_dim)).astype(self._np_dtype)
            ).to(self._device)
        self._actions = torch.from_numpy(
            np.zeros((1, self._acs_dim)).astype(self._np_dtype)
            ).to(self._device)
        self._rewards = torch.FloatTensor(
            np.array([])
            ).to(self._device)
        self._step_nums = torch.LongTensor(np.array([-1])).to(self._device)
        self._step_num = 0

        if prompt:
            prompt_obs = torch.from_numpy(
                np.vstack(prompt[0]).astype(self._np_dtype)
            ).to(self._device)
            self._observations = torch.cat(
                [self._observations, prompt_obs],
                dim=0
                )[-self._seq_len:]

            prompt_acs = torch.from_numpy(
                np.vstack(prompt[1]).astype(self._np_dtype)
            ).to(self._device)
            self._actions = torch.cat(
                [self._actions, prompt_acs],
                dim=0
                )[-self._seq_len:]

            prompt_rews = torch.FloatTensor(
                np.array(prompt[2])
                ).to(self._device)
            self._rewards = torch.cat(
                [self._rewards, prompt_rews],
                dim=0
                )[-self._seq_len:]

            prompt_step_nums = torch.LongTensor(
                np.array(prompt[3])
                ).to(self._device)
            self._step_nums = torch.cat(
                [self._step_nums, prompt_step_nums],
                dim=0
                )[-self._seq_len:]

    @torch.no_grad()
    def get_next_action(self,
                        observation,
                        prev_reward: Optional[float],
                        is_offline: Optional[bool] = False) -> np.ndarray:
        """Get next action

        Predict next action (`reset_context` method must be called
        before first usage of current method).

        Args:
            observation: new observation
            prev_reward: reward obtained before passed observation.
                None if it is first observation in epoch
            is_offline: flag if prediction should be in online or offline
                format, context is not updated in offline format.

        Returns:
            np.ndarray: predicted action
        """

        t_obs = torch.from_numpy(
                    observation.astype(self._np_dtype)
                    ).to(self._device).unsqueeze(0)
        if prev_reward is None:
            self._step_num = 0
            prev_reward = 0
        t_rew = torch.from_numpy(
                np.array([prev_reward]).astype(self._np_dtype)
                ).to(self._device)

        t_step_num = torch.from_numpy(
                np.array([self._step_num]).astype(int)
                ).to(self._device)

        inp_obs = torch.cat([self._observations, t_obs], dim=0)
        inp_acs = self._actions
        inp_rews = torch.cat([self._rewards, t_rew], dim=0)
        inp_snums = torch.cat([self._step_nums, t_step_num], dim=0)

        model_input = self.create_model_input(
            observations=inp_obs,
            actions=inp_acs,
            rewards=inp_rews,
            step_nums=inp_snums,
            task_name=self._task_name
        )

        with torch.amp.autocast("cuda", dtype=self._torch_dtype):
            model_action = self.get_action(
                        model_input)

        if not is_offline:
            self._observations = torch.cat(
                [self._observations, t_obs],
                dim=0)[-self._seq_len:]
            self._rewards = torch.cat(
                [self._rewards, t_rew],
                dim=0)[-self._seq_len+1:]
            t_act = torch.from_numpy(
                    model_action[0].astype(self._np_dtype)
                    ).to(self._device).unsqueeze(0)
            self._actions = torch.cat(
                [self._actions, t_act],
                dim=0)[-self._seq_len:]
            self._step_nums = torch.cat(
                [self._step_nums, t_step_num],
                dim=0)[-self._seq_len:]

        self._step_num += 1
        return model_action[0]
