import torch
import torch.nn as nn


class MultiModalLoss(nn.Module):
    def __init__(self):
        super(MultiModalLoss, self).__init__()

    def forward(self,
                predicted_trajectories,
                predicted_probabilities,
                ground_truth_trajectory,
                ground_truth_states_valid):
        ade_per_mode = self._compute_ade_per_trajectory(
            predicted_trajectories,
            ground_truth_trajectory,
            ground_truth_states_valid
        )
        return (
            self.weighted_nll_loss(
                ade_per_mode,
                predicted_probabilities,
                ground_truth_states_valid) +
            self.min_ade_loss(ade_per_mode,
                              ground_truth_states_valid) +
            self.diversity_Loss(predicted_trajectories)
        )

    def _compute_ade_per_trajectory(self, predicted_trajectories,
                                    ground_truth_trajectory,
                                    ground_truth_states_valid,
                                    use_yaw=False):
        """
        Computes the Average Displacement Error (ADE) per trajectory mode.

        Args:
            predicted_trajectories: 
            [batch_size, num_agents, num_trajectories, num_timesteps, 3] 
            ground_truth_trajectory: [batch_size, num_agents, num_timesteps, 3]
            ground_truth_states_valid: [batch_size, num_agents, num_timesteps] 
            storing validity information for each state.
            use_yaw: Whether to use yaw in the loss.

        Returns:
            Tensor: ADE per trajectory mode, with shape
            [batch_size, num_agents, num_trajectories].
        """
        # Calculate ADE for each mode
        # [batch_size, num_agents, num_trajectories, num_timesteps, 3]
        ade_diff = predicted_trajectories - \
            ground_truth_trajectory.unsqueeze(dim=-3)

        if use_yaw:
            # Handle yaw wrap-around
            # [batch_size, num_agents, num_trajectories, num_timesteps]
            yaw_diff = predicted_trajectories[..., 2] - \
                ground_truth_trajectory[..., 2].unsqueeze(dim=-2)
            yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(
                yaw_diff))  # Corrected yaw difference

            # Replace yaw with the corrected difference.
            ade_diff[..., 2] = yaw_diff
        else:
            # [batch_size, num_agents, num_trajectories, num_timesteps, 2]
            # Ignore yaw
            ade_diff = ade_diff[..., :2]

        # Apply state valid mask to the ade calculation.
        # [batch_size, num_agents, num_trajectories, num_timesteps, 2(or 3)]
        masked_diff = ade_diff * \
            ground_truth_states_valid.unsqueeze(dim=-1).unsqueeze(dim=-3)
        # [batch_size, num_agents, num_trajectories]
        ade_per_mode = torch.norm(masked_diff, dim=-1).sum(dim=-1) / \
            (ground_truth_states_valid.sum(dim=-1).unsqueeze(dim=-1) + 1e-8)

        return ade_per_mode

    def weighted_nll_loss(self, ade_per_mode,
                          predicted_probabilities,
                          ground_truth_states_valid):
        """
        Computes the weighted NLL loss for trajectory prediction.

        Args:
            ade_per_mode: [batch_size, num_agents, num_trajectories].
            predicted_probabilities: 
            [batch_size, num_agents, num_trajectories].
            ground_truth_states_valid: 
            [batch_size, num_agents, num_timesteps] storing validity 
            information for each state.

        Returns:
            Total loss (scalar).
        """
        # Negative Log-Likelihood (NLL) Loss
        # Adding a small epsilon for numerical stability
        # [batch_size, num_agents, num_trajectories]
        nll_loss = -torch.log(predicted_probabilities + 1e-8)

        # Calculate weighted NLL
        # [batch_size, num_agents]
        weighted_nll = (nll_loss * ade_per_mode).mean(dim=-1)

        # Calculate the mean loss across all active agents.
        # Compute invalid agents as one with no timestamp valid
        # [batch_size]
        valid_agents = ground_truth_states_valid.amax(dim=-1)
        valid_agent_count = valid_agents.sum()
        weighted_nll_loss = weighted_nll.sum() / valid_agent_count

        return weighted_nll_loss

    def min_ade_loss(self,
                     ade_per_mode,
                     ground_truth_states_valid):
        """
        Computes the minimum ADE loss for trajectory prediction.

        Args:
            ade_per_mode: [batch_size, num_agents, num_trajectories].
            ground_truth_states_valid: [batch_size, num_agents, num_timesteps] 
            storing validity information for each state.

        Returns:
            Total loss (scalar).
        """
        # Calculate minADE
        # [batch_size, num_agents]
        min_ade, _ = torch.min(ade_per_mode, dim=-1)

        # Calculate the mean loss across all active agents.
        # Compute invalid agents as one with no timestamp valid
        # [batch_size]
        valid_agents = ground_truth_states_valid.amax(dim=-1)
        valid_agent_count = valid_agents.sum()
        min_ade_loss = min_ade.sum() / valid_agent_count

        return min_ade_loss

    def diversity_Loss(self, predicted_trajectories):
        """
        Encourages diversity by penalizing similar trajectories.

        predicted_trajectories: (batch_size, num_agents, num_trajectories, timesteps, 3)
        """
        num_trajs = predicted_trajectories.size(dim=-3)
        diversity_loss = 0.0

        if num_trajs < 2:
            return diversity_loss

        for i in range(num_trajs):
            for j in range(i + 1, num_trajs):
                pairwise_dist = torch.norm(
                    predicted_trajectories[..., i, :, :2] -
                    predicted_trajectories[..., j, :, :2], dim=-1).mean()
                # Penalize trajectories that are too close
                diversity_loss += torch.exp(-pairwise_dist)

        diversity_loss /= (num_trajs * (num_trajs - 1)) / \
            2  # Normalize over number of pairs
        return diversity_loss
