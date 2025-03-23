import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    # k, v are of size [batch, num_heads, num_attending_inputs_kv, d_head]
    # q is of size [batch, num_heads, num_attending_inputs_q, d_head]

    # [batch, num_heads, num_attending_inputs_q, num_attending_inputs_kv]
    attention = torch.matmul(q, k.transpose(-2, -1))
    scale = q.size(dim=-1) ** 0.5
    attention = attention / scale
    # [batch, num_heads, num_attending_inputs_q, num_attending_inputs_kv]
    if mask is not None:
        attention = attention.masked_fill(mask == 0, float(-1e20))

    # [batch, num_heads, num_attending_inputs_q, num_attending_inputs_kv]
    attention = F.softmax(attention, dim=-1)

    # [batch, num_heads, num_attending_inputs_q, d_head]
    values = torch.matmul(attention, v)

    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, num_attending_inputs_qkv, _ = x.size()
        # [batch * num_attending_inputs_qkv, 3*d_model]
        qkv = self.qkv_linear(x)
        # [batch, num_attending_inputs_qkv, num_heads, 3*d_head]
        qkv = qkv.reshape(batch_size, num_attending_inputs_qkv,
                          self.num_heads, 3*self.head_dim)
        # [batch, num_heads, num_attending_inputs_qkv, 3*d_head]
        qkv = qkv.permute(0, 2, 1, 3)
        # [batch, num_heads, num_attending_inputs_qkv, d_head] each
        q, k, v = qkv.chunk(3, dim=-1)
        # [batch, num_heads, num_attending_inputs_qkv, d_head]
        values, _ = scaled_dot_product(q, k, v, mask.unsqueeze(dim=1))
        # [batch, num_attending_inputs_qkv, d_model]
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, num_attending_inputs_qkv, self.num_heads*self.head_dim)
        # [batch, num_attending_inputs_qkv, d_model]
        out = self.linear(values)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.kv_linear = nn.Linear(d_model, 2 * d_model)
        self.q_linear = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask):
        batch_size, num_attending_inputs_kv, _ = x.size()
        _, num_attending_inputs_q, _ = y.size()
        # [batch, num_attending_inputs_kv, 2*d_model]
        kv = self.kv_linear(x)
        # [batch, num_attending_inputs_q, d_model]
        q = self.q_linear(y)
        # [batch, num_attending_inputs_kv, num_heads, 2*d_head]
        kv = kv.reshape(batch_size, num_attending_inputs_kv,
                        self.num_heads, 2*self.head_dim)
        # [batch, num_heads, num_attending_inputs_kv, 2*d_head]
        kv = kv.permute(0, 2, 1, 3)
        # [batch, num_attending_inputs_q, num_heads, d_head]
        q = q.reshape(batch_size, num_attending_inputs_q,
                      self.num_heads, self.head_dim)
        # [batch, num_heads, num_attending_inputs_q, d_head]
        q = q.permute(0, 2, 1, 3)
        # [batch, num_heads, num_attending_inputs_kv, d_head]
        k, v = kv.chunk(2, dim=-1)
        # [batch, num_heads, num_attending_inputs_q, d_head]
        values, _ = scaled_dot_product(q, k, v, mask.unsqueeze(dim=1))
        # [batch, num_attending_inputs, d_model]
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, num_attending_inputs_q, self.num_heads*self.head_dim)
        # [batch, num_attending_inputs_q, d_model]
        out = self.linear(values)
        return out
