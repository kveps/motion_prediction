import torch
import torch.nn as nn
from models.transformer.attention import (
    MultiHeadAttention,
    MultiHeadCrossAttention,
)
from models.transformer.feed_forward import FeedForward
from models.transformer.layer_norm import LayerNorm


def _create_agent_agent_attention_mask(future_agents, future_agents_valid):
    batch_size, num_agents, num_future_timesteps, _ = future_agents.size()
    # Ensure mask is on the same device as future_agents
    device = future_agents.device
    
    # [batch_size*num_agents, num_future_timesteps, num_future_timesteps]
    time_attention_mask = future_agents_valid.reshape(
        batch_size * num_agents, num_future_timesteps
    ).unsqueeze(-1).repeat(1, 1, num_future_timesteps)
    # Create a mask such that agents don't look into the future
    # [batch_size*num_agents, num_future_timesteps, num_future_timesteps]
    zero_look_ahead_mask = torch.ones(
        batch_size*num_agents, num_future_timesteps, num_future_timesteps,
        device=device, dtype=time_attention_mask.dtype)
    zero_look_ahead_mask = torch.tril(zero_look_ahead_mask)
    # [batch_size*num_agents, num_future_timesteps, num_future_timesteps]
    time_attention_mask = time_attention_mask * zero_look_ahead_mask

    # [batch_size*num_future_timesteps, num_agents, num_agents]
    agent_attention_mask = future_agents_valid.swapaxes(1, 2).reshape(
        batch_size*num_future_timesteps, num_agents
    ).unsqueeze(-1).repeat(1, 1, num_agents).to(device)

    return time_attention_mask, agent_attention_mask


def _create_encoded_agent_future_agent_attention_mask(encoded_agents,
                                                      encoded_agents_valid,
                                                      future_agents,
                                                      future_agents_valid):
    batch_size, num_agents, num_encoded_timesteps, _ = encoded_agents.size()
    _, _, num_future_timesteps, _ = future_agents.size()
    # Ensure mask is on the same device as future_agents
    device = future_agents.device
    
    # [batch_size*num_agents, num_encoded_timesteps]
    encoded_agents_mask = encoded_agents_valid.reshape(
        batch_size*num_agents, num_encoded_timesteps)
    # [batch_size*num_agents, num_future_timesteps, num_encoded_timesteps]
    encoded_agents_mask = encoded_agents_mask.unsqueeze(-2).repeat(
        1, num_future_timesteps, 1
    )
    # [batch_size*num_agents, num_future_timesteps]
    future_agents_mask = future_agents_valid.reshape(
        batch_size*num_agents, num_future_timesteps)
    # [batch_size*num_agents, num_future_timesteps, num_encoded_timesteps]
    future_agents_mask = future_agents_mask.unsqueeze(-1).repeat(
        1, 1, num_encoded_timesteps
    )

    # [batch_size*num_agents, num_future_timesteps, num_encoded_timesteps]
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

    def forward(self, encoded_agents, encoded_agents_valid,
                future_agents, future_agents_valid):
        # [batch_size, num_agents, num_future_timesteps, d_model]
        residual_future_agents = future_agents.clone()

        # Future agent self attention
        #
        # Self attention on agents separately along the timestamp axis and
        # then the agents axis
        batch_size, num_agents, num_future_timesteps, _ = future_agents.size()
        time_attention_mask, future_agent_attention_mask = _create_agent_agent_attention_mask(
            future_agents, future_agents_valid
        )
        # # [batch_size, num_agents, num_future_timesteps, d_model]
        # future_agents = future_agents.reshape(
        #     batch_size, num_agents, num_future_timesteps, -1
        # )
        # Time attention on agents
        # [batch_size*num_agents, num_future_timesteps, d_model]
        time_attention = future_agents.reshape(
            batch_size * num_agents, num_future_timesteps, -1)
        # [batch_size*num_agents, num_future_timesteps, d_model]
        future_agents = self.self_attention(
            time_attention, mask=time_attention_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_future_timesteps, d_model]
        future_agents = future_agents.reshape(
            batch_size, num_agents, num_future_timesteps, -1
        )
        # Agent attention on agents
        # [batch_size*num_future_timesteps, num_agents, d_model]
        future_agent_attenion = future_agents.swapaxes(1, 2).reshape(
            batch_size*num_future_timesteps, num_agents, -1)
        future_agents = self.self_attention(
            future_agent_attenion, mask=future_agent_attention_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_future_timesteps, d_model]
        future_agents = future_agents.reshape(
            batch_size, num_future_timesteps, num_agents, -1).swapaxes(1, 2)

        # [batch_size, num_agents, num_future_timesteps, d_model]
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
        # [batch_size*num_agents, num_future_timesteps, d_model]
        future_agents = future_agents.reshape(
            batch_size*num_agents, num_future_timesteps, -1)
        # [batch_size*num_agents, num_future_timesteps, d_model]
        future_agents = self.cross_attention(
            encoded_agents, future_agents, mask=encoded_future_attention_mask)
        # Reshape agents back
        # [batch_size, num_agents, num_future_timesteps, d_model]
        future_agents = future_agents.reshape(
            batch_size, num_agents, num_future_timesteps, -1
        )

        # [batch_size, num_agents, num_future_timesteps, d_model]
        future_agents = self.dropout2(future_agents)
        future_agents = self.norm2(future_agents + residual_future_agents)

        # [batch_size, num_agents, num_future_timesteps, d_model]
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
                future_agents, future_agents_valid):
        for layer in self.layers:
            future_agents = layer(encoded_agents, past_agents_valid,
                                  future_agents, future_agents_valid)
        return future_agents
