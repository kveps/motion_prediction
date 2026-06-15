"""Per-mode endpoint anchors for the multi-modal decoder.

The prediction decoder produces K trajectory modes per agent. Before the
decoder runs, each (agent, mode) slot needs an initial query embedding — and
those queries come from a 2D endpoint anchor that gets projected to d_model
by the model's endpoint encoder.

This module is the single source of truth for "where does each mode's anchor
point come from?" Two sources are supported, both returning a
`[batch, agents, K, 2]` tensor of endpoints in SDC frame (the same frame the
model operates in):

  - BALLISTIC: per-agent, per-mode constant-turn-rate extrapolation.
        Mode i uses yaw rate omega_i, uniformly spaced in
        [-max_yaw_rate, +max_yaw_rate]. Diversity from physics; no offline
        data required. Diversity vanishes for stopped agents.

  - CENTROID:  K cluster centers (in agent-local frame) of training-set
        endpoints, transformed to SDC frame for each agent. Every agent
        sees the same K maneuvers, but anchored to its current position
        and heading. Provides stable semantic mode identity; no per-agent
        motion context. Requires pre-computed centroids on disk.

Run `compute_and_save_centroids(training_dataset, K, path)` once to produce
the centroids file consumed by CENTROID.
"""
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Anchor-type identifiers (string constants; pass as anchor_type=)
# ---------------------------------------------------------------------------
ANCHOR_BALLISTIC = 'ballistic'
ANCHOR_CENTROID  = 'centroid'

VALID_ANCHOR_TYPES = (ANCHOR_BALLISTIC, ANCHOR_CENTROID)

# Default agent-feature indices in the SDC-frame agent_continuous tensor.
# Layout: [x, y, sin_yaw, cos_yaw, vx, vy, sin_vel_yaw, cos_vel_yaw, length, width]
_IDX_X      = 0
_IDX_Y      = 1
_IDX_SIN_H  = 2
_IDX_COS_H  = 3
_IDX_VX     = 4
_IDX_VY     = 5


# ===========================================================================
# Public dispatcher
# ===========================================================================
def get_mode_anchor_endpoints(anchor_type, agents_continuous,
                              num_future_timesteps, num_modes,
                              centroids=None, max_yaw_rate=0.1, dt=0.1):
    """Compute per-agent, per-mode endpoint anchor points in SDC frame.

    Args:
        anchor_type: one of {ANCHOR_BALLISTIC, ANCHOR_CENTROID}.
        agents_continuous: [batch, agents, T_past, num_features] in SDC frame.
            Expects feature layout
            [x, y, sin_yaw, cos_yaw, vx, vy, ...]. Only the last past timestep
            is used to seed the anchor.
        num_future_timesteps: prediction horizon (used by ballistic to
            project forward).
        num_modes: K — number of trajectory modes.
        centroids: [K_centroids, 2] cluster centers in agent-local frame
            ([forward, left] meters). Required for CENTROID.
        max_yaw_rate: max |omega| in rad/s for the BALLISTIC mode.
        dt: timestep in seconds (default 0.1 for 10 Hz Waymo data).

    Returns:
        Tensor of shape [batch, agents, num_modes, 2] — (x_end, y_end) per
        (agent, mode) in SDC frame.
    """
    if anchor_type == ANCHOR_BALLISTIC:
        return _ballistic_anchor(agents_continuous, num_future_timesteps,
                                 num_modes, max_yaw_rate, dt)
    if anchor_type == ANCHOR_CENTROID:
        _require_centroids(centroids, anchor_type, num_modes)
        return _centroid_anchor(agents_continuous, centroids, num_modes)
    raise ValueError(
        f"Unknown anchor_type={anchor_type!r}. "
        f"Expected one of {VALID_ANCHOR_TYPES}.")


def _require_centroids(centroids, anchor_type, num_modes):
    if centroids is None:
        raise ValueError(
            f"anchor_type={anchor_type!r} requires `centroids`; got None. "
            "Run compute_and_save_centroids(...) and load them.")
    if centroids.size(0) < num_modes:
        raise ValueError(
            f"Need at least {num_modes} centroids for {num_modes} modes, "
            f"got {centroids.size(0)}.")


# ===========================================================================
# Anchor implementations
# ===========================================================================
def _ballistic_anchor(agents_continuous, num_future_timesteps, num_modes,
                      max_yaw_rate, dt):
    """K different constant-turn-rate endpoints per agent.

    Mode i uses yaw rate omega_i = max_yaw_rate * (2*i/(K-1) - 1) for K>=2
    (uniform spread). For K=1, omega=0 (pure constant-velocity).

    Closed-form constant-turn-rate motion model:
        x_end = x0 + speed * (sin(h_end) - sin(h0)) / omega         if omega != 0
        x_end = x0 + speed * cos(h0) * T                            if omega -> 0
        y_end = y0 + speed * (cos(h0) - cos(h_end)) / omega         if omega != 0
        y_end = y0 + speed * sin(h0) * T                            if omega -> 0
    """
    device, dtype = agents_continuous.device, agents_continuous.dtype

    x0 = agents_continuous[..., -1, _IDX_X]
    y0 = agents_continuous[..., -1, _IDX_Y]
    vx = agents_continuous[..., -1, _IDX_VX]
    vy = agents_continuous[..., -1, _IDX_VY]

    h0 = torch.atan2(vy, vx)
    speed = torch.sqrt(vx * vx + vy * vy + 1e-8)

    if num_modes == 1:
        omegas = torch.zeros(1, device=device, dtype=dtype)
    else:
        omegas = torch.linspace(-max_yaw_rate, max_yaw_rate,
                                num_modes, device=device, dtype=dtype)
    total_t = num_future_timesteps * dt

    x0_e    = x0.unsqueeze(-1)
    y0_e    = y0.unsqueeze(-1)
    h0_e    = h0.unsqueeze(-1)
    speed_e = speed.unsqueeze(-1)
    omegas_e = omegas.view(1, 1, num_modes)
    h_end = h0_e + omegas_e * total_t

    nonzero = omegas_e.abs() > 1e-6
    safe_omega = torch.where(nonzero, omegas_e, torch.ones_like(omegas_e))
    x_end = torch.where(
        nonzero,
        x0_e + speed_e * (torch.sin(h_end) - torch.sin(h0_e)) / safe_omega,
        x0_e + speed_e * torch.cos(h0_e) * total_t,
    )
    y_end = torch.where(
        nonzero,
        y0_e + speed_e * (torch.cos(h0_e) - torch.cos(h_end)) / safe_omega,
        y0_e + speed_e * torch.sin(h0_e) * total_t,
    )

    return torch.stack([x_end, y_end], dim=-1)  # [B, A, K, 2]


def _centroid_anchor(agents_continuous, centroids, num_modes):
    """Per-mode centroids transformed from agent-local frame to SDC frame.

    Every agent gets the same K centroid points, but each centroid is rotated
    by the agent's current heading and translated to the agent's current
    position — so the result is the K typical maneuver endpoints, anchored
    to where this specific agent is and which way it's facing.
    """
    K = num_modes
    centroids_kf = centroids[:K]  # [K, 2] in agent-local frame

    # Last observed state in SDC frame
    x0    = agents_continuous[..., -1, _IDX_X]      # [B, A]
    y0    = agents_continuous[..., -1, _IDX_Y]      # [B, A]
    sin_h = agents_continuous[..., -1, _IDX_SIN_H]  # [B, A]
    cos_h = agents_continuous[..., -1, _IDX_COS_H]  # [B, A]

    # Rotate K centroids by each agent's heading, then translate to position.
    # Result: [B, A, K, 2] in SDC frame.
    rotated   = _rotate_agent_local_to_sdc(centroids_kf, sin_h, cos_h)  # [B, A, K, 2]
    translated = _translate_to_position(rotated, x0, y0)                 # [B, A, K, 2]
    return translated


# ===========================================================================
# Coordinate-frame helpers
# ===========================================================================
def _rotate_agent_local_to_sdc(local_xy_K, sin_h, cos_h):
    """Rotate K 2D points from each agent's local frame into SDC frame.

    Agent-local frame: x = forward (along agent's heading), y = left.
    SDC frame: x = SDC forward, y = SDC left. The agent's heading in SDC
    frame is the rotation angle h with (sin_h, cos_h) = (sin h, cos h).

    Standard 2D rotation:
        sdc_x = cos_h * lx - sin_h * ly
        sdc_y = sin_h * lx + cos_h * ly

    Args:
        local_xy_K: [K, 2] points in agent-local frame.
        sin_h, cos_h: [B, A] each, agent heading components in SDC frame.

    Returns:
        [B, A, K, 2] points in SDC frame.
    """
    lx = local_xy_K[..., 0]  # [K]
    ly = local_xy_K[..., 1]  # [K]

    # Broadcast: [B, A, 1] * [K] -> [B, A, K]
    sin_h_e = sin_h.unsqueeze(-1)
    cos_h_e = cos_h.unsqueeze(-1)

    sdc_x = cos_h_e * lx - sin_h_e * ly
    sdc_y = sin_h_e * lx + cos_h_e * ly
    return torch.stack([sdc_x, sdc_y], dim=-1)  # [B, A, K, 2]


def _translate_to_position(rotated_xy, x0, y0):
    """Translate rotated K points by each agent's current SDC-frame position.

    Args:
        rotated_xy: [B, A, K, 2]
        x0, y0: [B, A] each.

    Returns:
        [B, A, K, 2]
    """
    return rotated_xy + torch.stack([x0, y0], dim=-1).unsqueeze(-2)


def sdc_endpoint_to_agent_local(x_curr, y_curr, sin_h, cos_h, x_end, y_end):
    """Inverse transform: convert a single endpoint from SDC frame to agent-local.

    Used during centroid computation: each training-set endpoint is converted
    to the corresponding agent's local frame before clustering, so the
    centroids are agent-invariant maneuver patterns.

    Inverse rotation (by -h):
        lx =  cos_h * dx + sin_h * dy
        ly = -sin_h * dx + cos_h * dy
    """
    dx = x_end - x_curr
    dy = y_end - y_curr
    lx =  cos_h * dx + sin_h * dy
    ly = -sin_h * dx + cos_h * dy
    return lx, ly


# ===========================================================================
# K-means centroid computation (offline, once per training-data version)
# ===========================================================================
def compute_centroids(dataset, num_centroids, max_samples=100_000, seed=0):
    """Cluster training-set endpoints in agent-local frame into K centroids.

    Iterates through `dataset`, collects per-agent (lx, ly) endpoint
    displacements relative to each agent's current pose, and runs K-means.

    Args:
        dataset: iterable yielding samples with keys
            agent_input_continuous, agent_target, agent_target_valid,
            agent_input_valid, tracks_to_predict.
            (Both MotionDataset and PreprocessedMotionDataset work.)
        num_centroids: K.
        max_samples: cap on collected endpoints — kmeans cost grows ~ N*K.
        seed: random seed for kmeans init.

    Returns:
        torch.Tensor [num_centroids, 2] of centroids in agent-local frame
        ([forward, left] meters).
    """
    # Lazy import — sklearn is only needed at training-prep time.
    from sklearn.cluster import KMeans  # noqa: WPS433

    endpoints = []
    samples_seen = 0
    for sample in dataset:
        samples_seen += 1
        agent_input  = sample['agent_input_continuous']   # [A, T_past, F]
        agent_target = sample['agent_target']             # [A, T_future, 4]
        target_valid = sample['agent_target_valid']       # [A, T_future]
        input_valid  = sample['agent_input_valid']        # [A, T_past]
        ttp          = sample['tracks_to_predict']        # [A]

        # Only consider agents we are asked to predict and that have a valid
        # current state and a valid endpoint.
        ttp_b = ttp.bool() if ttp.dtype != torch.bool else ttp
        curr_valid_b = input_valid[:, -1].bool()
        end_valid_b  = target_valid[:, -1].bool()
        keep = ttp_b & curr_valid_b & end_valid_b
        if not keep.any():
            continue

        x_curr = agent_input[keep, -1, _IDX_X]
        y_curr = agent_input[keep, -1, _IDX_Y]
        sin_h  = agent_input[keep, -1, _IDX_SIN_H]
        cos_h  = agent_input[keep, -1, _IDX_COS_H]
        x_end  = agent_target[keep, -1, 0]
        y_end  = agent_target[keep, -1, 1]

        lx, ly = sdc_endpoint_to_agent_local(x_curr, y_curr, sin_h, cos_h,
                                             x_end, y_end)
        for i in range(lx.numel()):
            endpoints.append((float(lx[i]), float(ly[i])))
            if len(endpoints) >= max_samples:
                break
        if len(endpoints) >= max_samples:
            break

    if not endpoints:
        raise RuntimeError("No valid endpoints collected — empty dataset?")

    E = np.asarray(endpoints, dtype=np.float32)
    print(f"Collected {len(E):,} endpoints from {samples_seen:,} scenes "
          f"for k-means.")

    km = KMeans(n_clusters=num_centroids, n_init=10, random_state=seed).fit(E)
    centroids = torch.from_numpy(km.cluster_centers_).float()  # [K, 2]
    print(f"K-means done. Inertia: {km.inertia_:.2f}")
    return centroids


def compute_and_save_centroids(dataset, num_centroids, output_path,
                                max_samples=100_000, seed=0):
    """Compute centroids and persist them to disk.

    Pretty-prints the result so the operator can sanity-check the maneuver
    semantics (forward = +x, left = +y in agent-local frame).
    """
    centroids = compute_centroids(dataset, num_centroids,
                                  max_samples=max_samples, seed=seed)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(centroids, output_path)
    print(f"Saved {num_centroids} centroids to {output_path}")
    print("Centroids in agent-local frame [forward (m), left (m)]:")
    for k in range(num_centroids):
        cx, cy = centroids[k, 0].item(), centroids[k, 1].item()
        print(f"  k={k}: ({cx:+8.2f}, {cy:+8.2f})")
    return centroids


def load_centroids(path):
    """Load centroids tensor from disk. Returns [K, 2] in agent-local frame."""
    return torch.load(path, weights_only=False)
