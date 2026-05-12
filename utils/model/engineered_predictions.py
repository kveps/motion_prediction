import torch


def ballistic_trajectories(agents_continuous, num_future_timesteps, dt=0.1):
    """Constant-velocity prediction from the last observed agent state.

    Agent feature layout: [x, y, sin_yaw, cos_yaw, vx, vy, ...]

    Args:
        agents_continuous: [batch, agents, timesteps, features]
        num_future_timesteps: number of future steps to generate
        dt: timestep in seconds (default 0.1 for 10 Hz)

    Returns:
        [batch, agents, num_future_timesteps, 4] — (x, y, sin_yaw, cos_yaw)
    """
    t = torch.arange(1, num_future_timesteps + 1,
                     device=agents_continuous.device,
                     dtype=agents_continuous.dtype) * dt

    x0  = agents_continuous[..., -1, 0]
    y0  = agents_continuous[..., -1, 1]
    sh  = agents_continuous[..., -1, 2]
    ch  = agents_continuous[..., -1, 3]
    vx  = agents_continuous[..., -1, 4]
    vy  = agents_continuous[..., -1, 5]

    bx  = x0.unsqueeze(-1) + vx.unsqueeze(-1) * t   # [B, A, T]
    by  = y0.unsqueeze(-1) + vy.unsqueeze(-1) * t
    bsh = sh.unsqueeze(-1).expand_as(bx)
    bch = ch.unsqueeze(-1).expand_as(bx)

    return torch.stack([bx, by, bsh, bch], dim=-1)  # [B, A, T, 4]
