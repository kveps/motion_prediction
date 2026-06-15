import torch.nn as nn
from models.transformer.attention import (
    MultiHeadAttention,
    MultiHeadCrossAttention,
    self_attn_capture,
    cross_attn_capture,
)
from models.transformer.feed_forward import FeedForward
from models.transformer.layer_norm import LayerNorm


def _create_agent_agent_attention_mask(agents, agents_valid):
    """Outer-product masks for time-axis and agent-axis self-attention.

    A (query, key) pair is unmasked iff both endpoints are valid.

    Returns:
        time_mask:  [B*A, T, T]  per (batch, agent): T queries x T keys
        agent_mask: [B*T, A, A]  per (batch, timestep): A queries x A keys
    """
    batch_size, num_agents, num_timesteps, _ = agents.size()
    device = agents.device

    time_v = agents_valid.reshape(batch_size * num_agents, num_timesteps).to(device)
    time_mask = time_v.unsqueeze(-1) * time_v.unsqueeze(-2)

    agent_v = agents_valid.swapaxes(1, 2).reshape(
        batch_size * num_timesteps, num_agents).to(device)
    agent_mask = agent_v.unsqueeze(-1) * agent_v.unsqueeze(-2)

    return time_mask, agent_mask


def _create_agent_static_road_attention_mask(agents, agents_valid,
                                             static_road, static_road_valid):
    """Cross-attention mask: per (batch, timestep), A agents x P static polylines.

    A polyline is considered valid if any of its points is valid (we max-pool
    points in pointnet, so a single valid point can carry information).
    """
    batch_size, num_agents, num_timesteps, _ = agents.size()
    _, num_static_rg, _, _ = static_road.size()
    device = agents.device

    # [B, P] — polyline valid if any point is valid
    polyline_v = static_road_valid.amax(-1)
    # [B*T, P]
    polyline_v = polyline_v.unsqueeze(1).repeat(1, num_timesteps, 1).reshape(
        batch_size * num_timesteps, num_static_rg)
    # [B*T, A]
    agent_v = agents_valid.swapaxes(1, 2).reshape(
        batch_size * num_timesteps, num_agents)
    # [B*T, A, P] = outer product
    mask = agent_v.unsqueeze(-1) * polyline_v.unsqueeze(-2)
    return mask.to(device)


def _create_agent_dynamic_road_attention_mask(agents, agents_valid,
                                              dynamic_road, dynamic_road_valid):
    """Cross-attention mask: per (batch, timestep), A agents x D dynamic objects."""
    batch_size, num_agents, num_timesteps, _ = agents.size()
    _, num_dynamic_rg, _, _ = dynamic_road.size()
    device = agents.device

    # [B*T, D]
    dyn_v = dynamic_road_valid.swapaxes(1, 2).reshape(
        batch_size * num_timesteps, num_dynamic_rg)
    # [B*T, A]
    agent_v = agents_valid.swapaxes(1, 2).reshape(
        batch_size * num_timesteps, num_agents)
    # [B*T, A, D]
    mask = agent_v.unsqueeze(-1) * dyn_v.unsqueeze(-2)
    return mask.to(device)


class EncoderLayer(nn.Module):
    """Single context-encoder layer.

    Five sub-layers, each followed by its own residual + LayerNorm:
      1. Time-axis self-attention   (per agent, attend across T past steps)
      2. Agent-axis self-attention  (per timestep, agents attend to each other)
      3. Cross-attention to static road polylines
      4. Cross-attention to dynamic road objects (traffic lights, etc.)
      5. Position-wise feed-forward network

    Each sub-layer has its own dedicated attention module — no weight sharing
    across axes or modalities, since each axis has different statistical
    structure and needs its own learned weights.
    """

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        # Separate attention modules per factored axis / per map modality.
        self.time_self_attention   = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.agent_self_attention  = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.static_road_cross_attention  = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.dynamic_road_cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)

        self.ffn = FeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)

        # One LayerNorm per sub-layer keeps activation magnitudes bounded
        # between sub-ops, so residual contributions are not numerically
        # swamped by upstream activation blow-ups.
        self.norm_time   = LayerNorm()
        self.norm_agent  = LayerNorm()
        self.norm_static = LayerNorm()
        self.norm_dyn    = LayerNorm()
        self.norm_ffn    = LayerNorm()

        self.dropout = nn.Dropout(p=drop_prob)

        self.captured = {}

    def forward(self, agents, agents_valid, static_road, static_road_valid,
                dynamic_road, dynamic_road_valid, capture=False):
        if capture:
            self.captured = {}
        captured = self.captured if capture else None

        B, A, T, D = agents.size()
        P_static = static_road.size(1)
        P_dyn = dynamic_road.size(1)

        time_mask, agent_mask = _create_agent_agent_attention_mask(agents, agents_valid)
        static_mask = _create_agent_static_road_attention_mask(
            agents, agents_valid, static_road, static_road_valid)
        dyn_mask = _create_agent_dynamic_road_attention_mask(
            agents, agents_valid, dynamic_road, dynamic_road_valid)

        # ---- Sub-layer 1: time-axis self-attention ----
        residual = agents
        x = agents.reshape(B * A, T, D)
        x = self_attn_capture(self.time_self_attention, x, time_mask,
                              captured, "time_self", B, A)
        agents = self.norm_time(
            self.dropout(x.reshape(B, A, T, D)) + residual)

        # ---- Sub-layer 2: agent-axis self-attention ----
        residual = agents
        x = agents.swapaxes(1, 2).reshape(B * T, A, D)
        x = self_attn_capture(self.agent_self_attention, x, agent_mask,
                              captured, "agent_self", B, T)
        agents = self.norm_agent(
            self.dropout(x.reshape(B, T, A, D).swapaxes(1, 2)) + residual)

        # ---- Sub-layer 3: cross-attention to static road polylines ----
        residual = agents
        kv = static_road.swapaxes(1, 2).reshape(B * T, P_static, D)
        q = agents.swapaxes(1, 2).reshape(B * T, A, D)
        x = cross_attn_capture(self.static_road_cross_attention, kv, q, static_mask,
                               captured, "agent_static_road", B, T)
        agents = self.norm_static(
            self.dropout(x.reshape(B, T, A, D).swapaxes(1, 2)) + residual)

        # ---- Sub-layer 4: cross-attention to dynamic road objects ----
        residual = agents
        kv = dynamic_road.swapaxes(1, 2).reshape(B * T, P_dyn, D)
        q = agents.swapaxes(1, 2).reshape(B * T, A, D)
        x = cross_attn_capture(self.dynamic_road_cross_attention, kv, q, dyn_mask,
                               captured, "agent_dynamic_road", B, T)
        agents = self.norm_dyn(
            self.dropout(x.reshape(B, T, A, D).swapaxes(1, 2)) + residual)

        # ---- Sub-layer 5: position-wise feed-forward ----
        residual = agents
        agents = self.norm_ffn(self.dropout(self.ffn(agents)) + residual)

        return agents


class ContextEncoder(nn.Module):
    def __init__(self, num_layers, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
            for _ in range(num_layers)
        ])

    def forward(self, agents, agents_mask, static_road, static_road_mask,
                dynamic_road, dynamic_road_mask, capture=False):
        for layer in self.layers:
            agents = layer(agents, agents_mask, static_road, static_road_mask,
                           dynamic_road, dynamic_road_mask, capture=capture)
        return agents
