import torch


def trajectory_prediction_multi_modal_loss(predicted_trajectories,
                                           predicted_probabilities,
                                           ground_truth_trajectory,
                                           ground_truth_states_valid,
                                           use_ade=True):
    """
    Computes the multi-modal loss for trajectory prediction, including yaw with wrap-around handling.

    Args:
        predicted_trajectories: Tensor of shape [batch_size, num_agents, num_trajectories, num_timesteps, 3] (x, y, yaw).
        predicted_probabilities: Tensor of shape [batch_size, num_agents,  num_trajectories].
        ground_truth_trajectory: Tensor of shape [batch_size, num_agents, num_timesteps, 3] (x, y, yaw).
        ground_truth_states_valid: Tensor of shape [batch_size, num_agents, num_timesteps] storing validity
            information for each state.
        use_ade: Boolean to determine whether to use ADE or FDE.

    Returns:
        Total loss (scalar).
    """
    # 1. Negative Log-Likelihood (NLL) Loss
    # Adding a small epsilon for numerical stability
    # [batch_size, num_agents, num_trajectories]
    nll_loss = -torch.log(predicted_probabilities + 1e-8)

    # Handle yaw wrap-around
    # [batch_size, num_agents, num_trajectories, num_timesteps]
    yaw_diff = predicted_trajectories[..., 2] - \
        ground_truth_trajectory[..., 2].unsqueeze(dim=-2)
    yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(
        yaw_diff))  # Corrected yaw difference

    # Calculate ADE for each mode (including corrected yaw)
    # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
    ade_diff = predicted_trajectories - \
        ground_truth_trajectory.unsqueeze(dim=-3)
    # Replace yaw with the corrected difference.
    ade_diff[..., 2] = yaw_diff

    # Apply state valid mask to the ade calculation.
    # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
    masked_diff = ade_diff * \
        ground_truth_states_valid.unsqueeze(dim=-1).unsqueeze(dim=-3)
    # [batch_size, num_agents, num_trajectories]
    ade_per_mode = torch.norm(masked_diff, dim=-1).sum(dim=-1) / (ground_truth_states_valid.sum(
        dim=-1).unsqueeze(dim=-1) + 1e-8)

    # [batch_size, num_agents, num_trajectories]
    yaw_fde_diff = yaw_diff[..., -1]

    # Calculate FDE for each mode (including corrected yaw)
    # [batch_size, num_agents, num_trajectories, 3]
    fde_diff = predicted_trajectories[..., -1, :] - \
        ground_truth_trajectory[..., -1, :].unsqueeze(dim=-2)
    fde_diff[..., 2] = yaw_fde_diff

    # Apply state valid mask to the fde calculation.
    # [batch_size, num_agents, num_trajectories, 3]
    masked_fde_diff = fde_diff * \
        ground_truth_states_valid[..., -1].unsqueeze(dim=-1).unsqueeze(dim=-1)
    # [batch_size, num_agents, num_trajectories]
    fde_per_mode = torch.norm(masked_fde_diff, dim=-1)

    # Calculate minADE and minFDE
    # [batch_size, num_agents]
    min_ade, _ = torch.min(ade_per_mode, dim=-1)
    min_fde, _ = torch.min(fde_per_mode, dim=-1)

    # Calculate weighted NLL based on minADE or minFDE
    # [batch_size, num_agents]
    weighted_nll = (
        nll_loss * (ade_per_mode if use_ade else fde_per_mode)).mean(dim=-1)

    # Calculate the mean loss across all active agents and timesteps.
    # [batch_size]
    valid_timestep_count = ground_truth_states_valid.sum()
    total_loss = (weighted_nll.sum() + (min_ade.sum()
                  if use_ade else min_fde.sum())) / valid_timestep_count

    return total_loss
