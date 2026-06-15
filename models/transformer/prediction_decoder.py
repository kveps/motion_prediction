import torch.nn as nn
from models.transformer.attention import (
    MultiHeadAttention,
    MultiHeadCrossAttention,
    self_attn_capture,
    cross_attn_capture,
)
from models.transformer.feed_forward import FeedForward
from models.transformer.layer_norm import LayerNorm


def _create_future_agent_agent_attention_mask(future_agents, future_agents_valid):
    """Outer-product mask: per (batch, mode), A queries x A keys.

    (q, k) is unmasked iff both endpoints are valid.
    """
    batch_size, num_agents, num_future_trajectories, _ = future_agents.size()
    device = future_agents.device
    agent_v = future_agents_valid.swapaxes(1, 2).reshape(
        batch_size * num_future_trajectories, num_agents).to(device)
    return agent_v.unsqueeze(-1) * agent_v.unsqueeze(-2)


def _create_encoded_agent_future_agent_attention_mask(encoded_agents,
                                                      encoded_agents_valid,
                                                      future_agents,
                                                      future_agents_valid):
    """Cross-attention mask: per (batch, agent), K mode queries x T_past keys.

    (q=mode, k=past_step) is unmasked iff both the agent's future is valid
    for that mode (always true here) and the agent's past at that timestep
    is valid.
    """
    batch_size, num_agents, num_encoded_timesteps, _ = encoded_agents.size()
    num_future_trajectories = future_agents.size(2)
    device = future_agents.device

    # [B*A, T_past]
    past_v = encoded_agents_valid.reshape(
        batch_size * num_agents, num_encoded_timesteps)
    # [B*A, K]
    future_v = future_agents_valid.reshape(
        batch_size * num_agents, num_future_trajectories)
    # [B*A, K, T_past]
    mask = future_v.unsqueeze(-1) * past_v.unsqueeze(-2)
    return mask.to(device)


class DecoderLayer(nn.Module):
    """Single prediction-decoder layer.

    Three sub-layers, each followed by its own residual + LayerNorm:
      1. Agent-axis self-attention  (per mode, future agents attend to each other)
      2. Cross-attention to encoded past (per (agent, mode) query, attend over past steps)
      3. Position-wise feed-forward network

    Note: mode self-attention was deliberately removed. It collapsed the K
    mode tokens to identical vectors by averaging, and standard motion
    prediction architectures (MTR, Wayformer, Scene Transformer) do not use
    it — K modes interact only through the loss.
    """

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.agent_self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.past_cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)

        self.norm_agent = LayerNorm()
        self.norm_past  = LayerNorm()
        self.norm_ffn   = LayerNorm()

        self.dropout = nn.Dropout(p=drop_prob)

        self.captured = {}

    def forward(self, encoded_agents, encoded_agents_valid,
                future_agents, future_agents_valid, capture=False):
        if capture:
            self.captured = {}
        captured = self.captured if capture else None

        B, A, K, D = future_agents.size()
        T_past = encoded_agents.size(2)

        agent_mask = _create_future_agent_agent_attention_mask(
            future_agents, future_agents_valid)
        past_mask = _create_encoded_agent_future_agent_attention_mask(
            encoded_agents, encoded_agents_valid,
            future_agents, future_agents_valid)

        # ---- Sub-layer 1: agent-axis self-attention ----
        # Per mode, the K parallel slices each run self-attention over A agents.
        residual = future_agents
        x = future_agents.swapaxes(1, 2).reshape(B * K, A, D)
        x = self_attn_capture(self.agent_self_attention, x, agent_mask,
                              captured, "agent_self", B, K)
        future_agents = self.norm_agent(
            self.dropout(x.reshape(B, K, A, D).swapaxes(1, 2)) + residual)

        # ---- Sub-layer 2: cross-attention to encoded past ----
        # For each agent, the K mode queries attend over that agent's past timeline.
        residual = future_agents
        kv = encoded_agents.reshape(B * A, T_past, D)
        q = future_agents.reshape(B * A, K, D)
        x = cross_attn_capture(self.past_cross_attention, kv, q, past_mask,
                               captured, "mode_past", B, A)
        future_agents = self.norm_past(
            self.dropout(x.reshape(B, A, K, D)) + residual)

        # ---- Sub-layer 3: position-wise feed-forward ----
        residual = future_agents
        future_agents = self.norm_ffn(self.dropout(self.ffn(future_agents)) + residual)

        return future_agents


class PredictionDecoder(nn.Module):
    def __init__(self, num_layers, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, ffn_hidden=ffn_hidden,
                         num_heads=num_heads, drop_prob=drop_prob)
            for _ in range(num_layers)
        ])

    def forward(self, encoded_agents, past_agents_valid,
                future_agents, future_agents_valid, capture=False):
        for layer in self.layers:
            future_agents = layer(encoded_agents, past_agents_valid,
                                  future_agents, future_agents_valid,
                                  capture=capture)
        return future_agents
