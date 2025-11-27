import torch.nn as nn
from models.transformer.attention import (
    MultiHeadAttention,
    MultiHeadCrossAttention,
)
from models.transformer.feed_forward import FeedForward
from models.transformer.layer_norm import LayerNorm


def _create_agent_agent_attention_mask(agents, agents_valid):
    batch_size, num_agents, num_timesteps, _ = agents.size()
    # Ensure mask is on the same device as agents
    device = agents.device
    
    # [batch_size*num_agents, num_timesteps, num_timesteps]
    time_attention_mask = agents_valid.reshape(
        batch_size * num_agents, num_timesteps
    ).unsqueeze(-1).repeat(1, 1, num_timesteps).to(device)

    # [batch_size*num_timesteps, num_agents, num_agents]
    agent_attention_mask = agents_valid.swapaxes(1, 2).reshape(
        batch_size * num_timesteps, num_agents,
    ).unsqueeze(-1).repeat(1, 1, num_agents).to(device)

    return time_attention_mask, agent_attention_mask


def _create_agent_static_road_attention_mask(agents, agents_valid,
                                             static_road, static_road_valid):
    batch_size, num_agents, num_timesteps, _ = agents.size()
    _, num_static_rg, _, _ = static_road.size()
    # Ensure mask is on the same device as agents
    device = agents.device
    
    # Max ensures that even if one polyline point is valid, the whole polyline
    # is valid. This might be okay since we max pool in the points dimension
    #  in pointnet.
    # [batch_size, num_static_rg]
    static_road_mask = static_road_valid.amax(-1)
    # [batch_size, num_timesteps, num_static_rg]
    static_road_mask = static_road_mask.unsqueeze(1).repeat(
        1, num_timesteps, 1
    )
    # [batch_size*num_timesteps, num_static_rg]
    static_road_mask = static_road_mask.reshape(
        batch_size*num_timesteps, num_static_rg)
    # [batch_size*num_timesteps, num_agents, num_static_rg]
    static_road_mask = static_road_mask.unsqueeze(1).repeat(
        1, num_agents, 1)
    # [batch_size*num_timesteps, num_agents]
    agent_mask = agents_valid.swapaxes(1, 2).reshape(
        batch_size*num_timesteps, num_agents)
    # [batch_size*num_timesteps, num_agents, num_static_rg]
    agent_mask = agent_mask.unsqueeze(-1).repeat(
        1, 1, num_static_rg)
    # [batch_size*num_timesteps, num_static_rg, num_agents]
    static_road_agent_mask = (static_road_mask * agent_mask).to(device)

    return static_road_agent_mask


def _create_agent_dynamic_road_attention_mask(agents, agents_valid,
                                              dynamic_road,
                                              dynamic_road_valid):
    batch_size, num_agents, num_timesteps, _ = agents.size()
    _, num_dynamic_rg, _, _ = dynamic_road.size()
    # Ensure mask is on the same device as agents
    device = agents.device
    
    # Create a mask for dynamic road and agents
    # [batch_size*num_timesteps, num_dynamic_rg]
    dynamic_road_mask = dynamic_road_valid.swapaxes(1, 2).reshape(
        batch_size*num_timesteps, num_dynamic_rg)
    # [batch_size*num_timesteps, num_agents, num_dynamic_rg]
    dynamic_road_mask = dynamic_road_mask.unsqueeze(1).repeat(
        1, num_agents, 1)
    # [batch_size*num_timesteps, num_agents]
    agent_mask = agents_valid.swapaxes(1, 2).reshape(
        batch_size*num_timesteps, num_agents)
    # [batch_size*num_timesteps, num_agents, num_dynamic_rg]
    agent_mask = agent_mask.unsqueeze(2).repeat(
        1, 1, num_dynamic_rg)
    # [batch_size*num_timesteps, num_dynamic_rg, num_agents]
    dynamic_road_agent_mask = (dynamic_road_mask * agent_mask).to(device)

    return dynamic_road_agent_mask


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads)
        self.cross_attention = MultiHeadCrossAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.norm1 = LayerNorm()
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = FeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm()
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, agents, agents_valid, static_road,
                static_road_valid, dynamic_road, dynamic_road_valid):
        # [batch_size, num_agents, num_timesteps, d_model]
        residual_agents = agents.clone()

        # Agent self attention
        #
        # Self attention on agents separately along the timestamp axis and
        # then the agents axis
        batch_size, num_agents, num_timesteps, _ = agents.size()
        time_attention_mask, agent_attention_mask = _create_agent_agent_attention_mask(
            agents, agents_valid
        )
        # Time attention on agents
        time_attention = agents.reshape(
            batch_size * num_agents, num_timesteps, -1)
        # [batch_size*num_agents, num_timesteps, d_model]
        agents = self.self_attention(time_attention, mask=time_attention_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_timesteps, d_model]
        agents = agents.reshape(
            batch_size, num_agents, num_timesteps, -1
        )
        # Agent attention on agents
        # [batch_size*num_timesteps, num_agents,d_model]
        agent_attenion = agents.swapaxes(1, 2).reshape(
            batch_size*num_timesteps, num_agents, -1)
        agents = self.self_attention(agent_attenion, mask=agent_attention_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_timesteps, d_model]
        agents = agents.reshape(
            batch_size, num_timesteps, num_agents, -1).swapaxes(1, 2)

        # Cross attention on agents and static road
        #
        _, num_static_rg, _, _ = static_road.size()
        # Create a mask for static road and agents
        # [batch_size, num_timesteps, num_static_rg]
        static_road_agent_mask = _create_agent_static_road_attention_mask(
            agents, agents_valid, static_road, static_road_valid)
        # [batch_size*num_timesteps, num_static_rg, d_model]
        static_road = static_road.swapaxes(1, 2).reshape(
            batch_size*num_timesteps, num_static_rg, -1
        )
        # [batch_size*num_timesteps, num_agents, d_model]
        agents = agents.swapaxes(1, 2).reshape(
            batch_size*num_timesteps, num_agents, -1)
        # [batch_size*num_timesteps, num_agents, d_model]
        agents = self.cross_attention(
            static_road, agents, mask=static_road_agent_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_timesteps, d_model]
        agents = agents.reshape(
            batch_size, num_timesteps, num_agents, -1
        ).swapaxes(1, 2)

        # Cross attention on agents and dynamic road
        #
        _, num_dynamic_rg, _, _ = dynamic_road.size()
        # Create a mask for dynamic road and agents
        # [batch_size*num_timesteps, num_dynamic_rg]
        dynamic_road_agent_mask = _create_agent_dynamic_road_attention_mask(
            agents, agents_valid, dynamic_road, dynamic_road_valid)
        # [batch_size*num_timesteps, num_dynamic_rg, d_model]
        dynamic_road = dynamic_road.swapaxes(1, 2).reshape(
            batch_size*num_timesteps, num_dynamic_rg, -1
        )
        # [batch_size*num_timesteps, num_agents, d_model]
        agents = agents.swapaxes(1, 2).reshape(
            batch_size*num_timesteps, num_agents, -1)
        # [batch_size*num_timesteps, num_agents, d_model]
        agents = self.cross_attention(
            dynamic_road, agents, mask=dynamic_road_agent_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_timesteps, d_model]
        agents = agents.reshape(
            batch_size, num_timesteps, num_agents, -1
        ).swapaxes(1, 2)

        # [batch_size, num_agents, num_timesteps, d_model]
        agents = self.dropout1(agents)
        agents = self.norm1(agents + residual_agents)

        # [batch_size, num_agents, num_timesteps, d_model]
        residual_agents = agents.clone()
        agents = self.ffn(agents)
        agents = self.dropout2(agents)
        agents = self.norm2(agents + residual_agents)

        return agents


class ContextEncoder(nn.Module):
    def __init__(self, num_layers, d_model, ffn_hidden, num_heads, drop_prob):
        super(ContextEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(
            d_model, ffn_hidden, num_heads, drop_prob)
            for _ in range(num_layers)])

    def forward(self, agents, agents_mask, static_road, static_road_mask,
                dynamic_road, dynamic_road_mask):
        for layer in self.layers:
            agents = layer(agents, agents_mask, static_road,
                           static_road_mask, dynamic_road, dynamic_road_mask)
        return agents
