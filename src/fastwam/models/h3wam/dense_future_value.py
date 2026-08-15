"""FACT-style dense future/value expert over frozen H3 observation features."""

from __future__ import annotations

import torch
from torch import nn


class DenseTemporalFutureValueModel(nn.Module):
    """Jointly predict future H3, future state and continuous lower-is-better cost.

    The module is intentionally small relative to H3.  Candidate actions are
    detached at its boundary, so dense value training cannot update the action
    generator that proposed them.  Module names match the proven C38 temporal
    consequence expert where possible, allowing an audited partial restore.
    """

    def __init__(self, *, state_dim: int = 8, action_dim: int = 7,
                 action_horizon: int = 32, actions_per_latent: int = 4,
                 h3_feature_dim: int = 5376, target_dim: int = 256,
                 hidden_dim: int = 256, num_heads: int = 8,
                 feature_input_scale: float = 0.009606920816877307,
                 projection_seed: int = 20260815) -> None:
        super().__init__()
        if action_horizon % actions_per_latent or hidden_dim % num_heads:
            raise ValueError("invalid temporal/value dimensions")
        self.state_dim=state_dim; self.action_dim=action_dim; self.action_horizon=action_horizon
        self.actions_per_latent=actions_per_latent; self.action_tokens=action_horizon//actions_per_latent
        self.h3_feature_dim=h3_feature_dim; self.target_dim=target_dim
        self.feature_input_scale=feature_input_scale; self.projection_seed=projection_seed
        g=torch.Generator(device="cpu").manual_seed(projection_seed)
        self.register_buffer("fixed_projection",torch.randn(h3_feature_dim,target_dim,generator=g)/h3_feature_dim**.5)
        self.state_encoder=nn.Sequential(nn.Linear(state_dim,hidden_dim),nn.SiLU(),nn.LayerNorm(hidden_dim))
        self.visual_encoder=nn.Sequential(nn.Linear(target_dim,hidden_dim),nn.SiLU(),nn.LayerNorm(hidden_dim))
        self.action_encoder=nn.Sequential(nn.Linear(actions_per_latent*action_dim,hidden_dim),nn.SiLU(),nn.LayerNorm(hidden_dim))
        self.action_position=nn.Parameter(torch.randn(1,self.action_tokens,hidden_dim,generator=g)/hidden_dim**.5)
        self.future_query=nn.Parameter(torch.randn(1,1,hidden_dim,generator=g)/hidden_dim**.5)
        self.context_norm=nn.LayerNorm(hidden_dim)
        self.future_attention=nn.MultiheadAttention(hidden_dim,num_heads,batch_first=True)
        self.predictor=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,target_dim))
        self.future_state_decoder=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,state_dim))
        self.value_decoder=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))

    def project_features(self,h3_features:torch.Tensor)->torch.Tensor:
        if h3_features.ndim==4 and h3_features.shape[1]==1: h3_features=h3_features[:,0]
        if h3_features.ndim!=3 or h3_features.shape[-1]!=self.h3_feature_dim: raise ValueError("invalid H3 feature shape")
        return h3_features.float().mean(1)*self.feature_input_scale@self.fixed_projection.to(h3_features.device)

    def forward_projected(self,current_proprio:torch.Tensor,current_target:torch.Tensor,candidate_actions:torch.Tensor)->dict[str,torch.Tensor]:
        b=current_proprio.shape[0]
        if current_proprio.shape!=(b,self.state_dim) or current_target.shape!=(b,self.target_dim) or candidate_actions.shape!=(b,self.action_horizon,self.action_dim): raise ValueError("invalid dense value input shape")
        actions=candidate_actions.detach().float().reshape(b,self.action_tokens,self.actions_per_latent*self.action_dim)
        state=self.state_encoder(current_proprio.float()).unsqueeze(1); visual=self.visual_encoder(current_target.float()).unsqueeze(1)
        action=self.action_encoder(actions)+self.action_position
        context=self.context_norm(torch.cat((state,visual,action),1)); query=self.future_query.expand(b,-1,-1)+state+visual
        future,_=self.future_attention(query,context,context,need_weights=False); latent=future[:,0]
        return {"future_h3":current_target+self.predictor(latent),"future_state":current_proprio.float()+self.future_state_decoder(latent),"value":self.value_decoder(latent)[:,0]}

    def forward(self,current_proprio:torch.Tensor,h3_features:torch.Tensor,candidate_actions:torch.Tensor)->dict[str,torch.Tensor]:
        return self.forward_projected(current_proprio,self.project_features(h3_features),candidate_actions)


__all__=["DenseTemporalFutureValueModel"]
