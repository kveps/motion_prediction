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


def mode_diverse_ballistic_endpoints(agents_continuous, num_future_timesteps,
                                     num_modes, max_yaw_rate=0.1, dt=0.1):
    """K ballistic endpoints, each generated with a different constant yaw rate.

    Provides initialization-time diversity for multi-mode prediction: each
    mode starts with a slightly different curvature so the K decoder queries
    are not identical at warm-start.

    Mode i uses yaw rate omega_i = max_yaw_rate * (2*i/(K-1) - 1) for K>=2
    (uniform spread from -max_yaw_rate to +max_yaw_rate). For K=1, omega=0.

    Args:
        agents_continuous: [batch, agents, timesteps, features]
        num_future_timesteps: int — projection horizon
        num_modes: K
        max_yaw_rate: max |omega| in rad/s (default 0.1 → ~46° turn over 8s at the extremes)
        dt: timestep in seconds

    Returns:
        [batch, agents, num_modes, 2] — (x_end, y_end) per mode
    """
    device = agents_continuous.device
    dtype = agents_continuous.dtype

    # Last observed state
    x0 = agents_continuous[..., -1, 0]                  # [B, A]
    y0 = agents_continuous[..., -1, 1]
    vx = agents_continuous[..., -1, 4]
    vy = agents_continuous[..., -1, 5]

    # Initial heading from velocity vector
    h0 = torch.atan2(vy, vx)                            # [B, A]
    speed = torch.sqrt(vx * vx + vy * vy + 1e-8)        # [B, A]

    if num_modes == 1:
        omegas = torch.zeros(1, device=device, dtype=dtype)
    else:
        omegas = torch.linspace(-max_yaw_rate, max_yaw_rate,
                                num_modes, device=device, dtype=dtype)

    total_time = num_future_timesteps * dt              # scalar

    # Broadcast: x0,y0,h0,speed -> [B, A, 1]; omegas -> [1, 1, K]
    x0_e = x0.unsqueeze(-1)
    y0_e = y0.unsqueeze(-1)
    h0_e = h0.unsqueeze(-1)
    speed_e = speed.unsqueeze(-1)
    omegas_e = omegas.view(1, 1, num_modes)
    h_end = h0_e + omegas_e * total_time

    # Closed-form constant-turn-rate motion model:
    #   x_end = x0 + speed * (sin(h_end) - sin(h0)) / omega        if omega != 0
    #   x_end = x0 + speed * cos(h0) * T                           in the omega -> 0 limit
    #   y_end = y0 + speed * (cos(h0) - cos(h_end)) / omega        if omega != 0
    #   y_end = y0 + speed * sin(h0) * T                           in the omega -> 0 limit
    nonzero = omegas_e.abs() > 1e-6
    safe_omega = torch.where(nonzero, omegas_e, torch.ones_like(omegas_e))

    x_end = torch.where(
        nonzero,
        x0_e + speed_e * (torch.sin(h_end) - torch.sin(h0_e)) / safe_omega,
        x0_e + speed_e * torch.cos(h0_e) * total_time,
    )
    y_end = torch.where(
        nonzero,
        y0_e + speed_e * (torch.cos(h0_e) - torch.cos(h_end)) / safe_omega,
        y0_e + speed_e * torch.sin(h0_e) * total_time,
    )

    return torch.stack([x_end, y_end], dim=-1)          # [B, A, K, 2]
