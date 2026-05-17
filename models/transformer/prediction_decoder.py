import torch
import torch.nn as nn
from models.transformer.attention import (
    MultiHeadAttention,
    MultiHeadCrossAttention,
)
from models.transformer.feed_forward import FeedForward
from models.transformer.layer_norm import LayerNorm


def _create_future_agent_agent_attention_mask(future_agents, future_agents_valid):
    batch_size, num_agents, num_future_trajectories, _ = future_agents.size()
    device = future_agents.device

    # Outer-product masks: (q, k) unmasked iff both endpoints are valid.

    # Agent mask: per (batch, trajectory), A_query x A_key
    agent_v = future_agents_valid.swapaxes(1, 2).reshape(
        batch_size * num_future_trajectories, num_agents).to(device)
    agent_attention_mask = agent_v.unsqueeze(-1) * agent_v.unsqueeze(-2)

    # Trajectory (mode) mask: per (batch, agent), K_query x K_key
    traj_v = future_agents_valid.reshape(
        batch_size * num_agents, num_future_trajectories).to(device)
    trajectory_attention_mask = traj_v.unsqueeze(-1) * traj_v.unsqueeze(-2)

    return agent_attention_mask, trajectory_attention_mask

def _create_encoded_agent_future_agent_attention_mask(encoded_agents,
                                                      encoded_agents_valid,
                                                      future_agents,
                                                      future_agents_valid):
    batch_size, num_agents, num_encoded_timesteps, _ = encoded_agents.size()
    num_future_trajectories = future_agents.size(2)
    # Ensure mask is on the same device as future_agents
    device = future_agents.device
    
    # [batch_size*num_agents, num_future_trajectories, num_encoded_timesteps]
    encoded_agents_mask = encoded_agents_valid.reshape(
        batch_size*num_agents, num_encoded_timesteps).unsqueeze(-2).repeat(
            1, num_future_trajectories, 1
        )
    # [batch_size*num_agents, num_future_trajectories]
    future_agents_mask = future_agents_valid.reshape(
        batch_size*num_agents, num_future_trajectories)
    # [batch_size*num_agents, num_future_trajectories, num_encoded_timesteps]
    future_agents_mask = future_agents_mask.unsqueeze(-1).repeat(
        1, 1, num_encoded_timesteps
    )

    # [batch_size*num_agents, num_future_trajectories, num_encoded_timesteps]
    return (encoded_agents_mask * future_agents_mask).to(device)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNorm()
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.cross_attention = MultiHeadCrossAttention(
            d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNorm()
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = FeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm()
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.captured = {}

    def forward(self, encoded_agents, encoded_agents_valid,
                future_agents, future_agents_valid, capture=False):
        if capture:
            self.captured = {}

        # [batch_size, num_agents, num_future_timesteps, d_model]
        residual_future_agents = future_agents.clone()

        # Future agent self attention
        #
        # Self attention on agents separately along the timestamp axis and
        # then the agents axis
        batch_size, num_agents, num_future_trajectories, _ = future_agents.size()
        future_agent_attention_mask, future_trajectory_attention_mask = _create_future_agent_agent_attention_mask(
            future_agents, future_agents_valid
        )
        # Agent attention on agents
        # [batch_size*num_future_trajectories, num_agents, d_model]
        future_agent_attention = future_agents.swapaxes(1, 2).reshape(
            batch_size*num_future_trajectories, num_agents, -1)
        if capture:
            future_agents, agent_self_attn = self.self_attention(
                future_agent_attention, mask=future_agent_attention_mask, return_attention=True)
            # [batch*K, h, A, A] -> [batch, K, h, A, A]
            self.captured["agent_self"] = agent_self_attn.reshape(
                batch_size, num_future_trajectories, *agent_self_attn.shape[1:]
            ).detach()
        else:
            future_agents = self.self_attention(
                future_agent_attention, mask=future_agent_attention_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_future_trajectories, d_model]
        future_agents = future_agents.reshape(
            batch_size, num_agents, num_future_trajectories, -1)
        # Trajectory attention on agents
        # [batch_size*num_agents, num_future_trajectories, d_model]
        future_trajectory_attention = future_agents.reshape(
            batch_size*num_agents, num_future_trajectories, -1)
        if capture:
            future_agents, mode_self_attn = self.self_attention(
                future_trajectory_attention, mask=future_trajectory_attention_mask, return_attention=True)
            # [batch*A, h, K, K] -> [batch, A, h, K, K]
            self.captured["mode_self"] = mode_self_attn.reshape(
                batch_size, num_agents, *mode_self_attn.shape[1:]
            ).detach()
        else:
            future_agents = self.self_attention(
                future_trajectory_attention, mask=future_trajectory_attention_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_future_trajectories, d_model]
        future_agents = future_agents.reshape(
            batch_size, num_agents, num_future_trajectories, -1
        )

        # [batch_size, num_agents, num_future_trajectories, d_model]
        future_agents = self.dropout1(future_agents)
        future_agents = self.norm1(future_agents + residual_future_agents)
        residual_future_agents = future_agents.clone()

        # Cross attention on future agents and past agents
        #
        _, _, num_encoded_timesteps, _ = encoded_agents.size()
        encoded_future_attention_mask = _create_encoded_agent_future_agent_attention_mask(
            encoded_agents, encoded_agents_valid,
            future_agents, future_agents_valid,
        )
        # [batch_size*num_agents, num_encoded_timesteps, d_model]
        encoded_agents = encoded_agents.reshape(
            batch_size*num_agents, num_encoded_timesteps, -1)
        # [batch_size*num_agents, num_future_trajectories, d_model]
        future_agents = future_agents.reshape(
            batch_size*num_agents, num_future_trajectories, -1)
        # [batch_size*num_agents, num_future_trajectories, d_model]
        if capture:
            future_agents, mode_past_attn = self.cross_attention(
                encoded_agents, future_agents, mask=encoded_future_attention_mask, return_attention=True)
            # [batch*A, h, K, T] -> [batch, A, h, K, T]
            self.captured["mode_past"] = mode_past_attn.reshape(
                batch_size, num_agents, *mode_past_attn.shape[1:]
            ).detach()
        else:
            future_agents = self.cross_attention(
                encoded_agents, future_agents, mask=encoded_future_attention_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_future_trajectories, d_model]
        future_agents = future_agents.reshape(
            batch_size, num_agents, num_future_trajectories, -1
        )

        # [batch_size, num_agents, num_future_trajectories, d_model]
        future_agents = self.dropout2(future_agents)
        future_agents = self.norm2(future_agents + residual_future_agents)

        # [batch_size, num_agents, num_future_trajectories, d_model]
        residual_future_agents = future_agents.clone()
        future_agents = self.ffn(future_agents)
        future_agents = self.dropout3(future_agents)
        future_agents = self.norm3(future_agents + residual_future_agents)

        return future_agents


class PredictionDecoder(nn.Module):
    def __init__(self, num_layers, d_model, ffn_hidden, num_heads, drop_prob):
        super(PredictionDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  num_heads=num_heads,
                                                  drop_prob=drop_prob)
                                    for _ in range(num_layers)])

    def forward(self, encoded_agents, past_agents_valid,
                future_agents, future_agents_valid, capture=False):
        for layer in self.layers:
            future_agents = layer(encoded_agents, past_agents_valid,
                                  future_agents, future_agents_valid,
                                  capture=capture)
        return future_agents
