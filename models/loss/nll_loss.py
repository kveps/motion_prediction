import torch
import torch.nn as nn


class NLL_Loss(nn.Module):
    def __init__(self):
        super(NLL_Loss, self).__init__()

    def forward(self,
                predicted_trajectories,
                predicted_probabilities,
                ground_truth_trajectory,
                ground_truth_states_valid,
                tracks_to_predict):
        ade_per_mode = self._compute_ade_per_trajectory(
            predicted_trajectories,
            ground_truth_trajectory,
            ground_truth_states_valid
        )
        return (
            self.weighted_nll_loss(
                ade_per_mode,
                predicted_probabilities,
                ground_truth_states_valid,
                tracks_to_predict) +
            self.min_ade_loss(ade_per_mode,
                              ground_truth_states_valid,
                              tracks_to_predict) +
            self.diversity_Loss(predicted_trajectories,
                               ground_truth_states_valid,
                               tracks_to_predict)
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
                          ground_truth_states_valid,
                          tracks_to_predict):
        """
        Computes the weighted NLL loss for trajectory prediction.
        Only counts loss for valid timesteps and predicted tracks.

        Args:
            ade_per_mode: [batch_size, num_agents, num_trajectories].
            predicted_probabilities: 
            [batch_size, num_agents, num_trajectories].
            ground_truth_states_valid: 
            [batch_size, num_agents, num_timesteps] storing validity 
            information for each state.
            tracks_to_predict: [batch_size, num_agents] boolean mask for agents to predict.

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

        # Calculate the mean loss only for valid timesteps and agents.
        # [batch_size, num_agents, num_timesteps] -> [batch_size, num_agents]
        valid_per_agent = ground_truth_states_valid.sum(dim=-1)
        
        # Only include agents that have at least one valid timestep and are marked to predict
        # [batch_size, num_agents]
        valid_agents_mask = (valid_per_agent > 0) & tracks_to_predict.bool()
        
        # Sum weighted_nll only for valid agents
        if valid_agents_mask.sum() > 0:
            weighted_nll_loss = weighted_nll[valid_agents_mask].sum() / valid_agents_mask.sum()
        else:
            weighted_nll_loss = torch.tensor(0.0, device=weighted_nll.device)

        return weighted_nll_loss

    def min_ade_loss(self,
                     ade_per_mode,
                     ground_truth_states_valid,
                     tracks_to_predict):
        """
        Computes the minimum ADE loss for trajectory prediction.
        Only counts loss for valid timesteps and predicted tracks.

        Args:
            ade_per_mode: [batch_size, num_agents, num_trajectories].
            ground_truth_states_valid: [batch_size, num_agents, num_timesteps] 
            storing validity information for each state.
            tracks_to_predict: [batch_size, num_agents] boolean mask for agents to predict.

        Returns:
            Total loss (scalar).
        """
        # Calculate minADE
        # [batch_size, num_agents]
        min_ade, _ = torch.min(ade_per_mode, dim=-1)

        # Calculate the mean loss only for valid timesteps and agents.
        # [batch_size, num_agents, num_timesteps] -> [batch_size, num_agents]
        valid_per_agent = ground_truth_states_valid.sum(dim=-1)
        
        # Only include agents that have at least one valid timestep and are marked to predict
        # [batch_size, num_agents]
        valid_agents_mask = (valid_per_agent > 0) & tracks_to_predict.bool()
        
        # Sum min_ade only for valid agents
        if valid_agents_mask.sum() > 0:
            min_ade_loss = min_ade[valid_agents_mask].sum() / valid_agents_mask.sum()
        else:
            min_ade_loss = torch.tensor(0.0, device=min_ade.device)

        return min_ade_loss

    def diversity_Loss(self, predicted_trajectories, ground_truth_states_valid, tracks_to_predict):
        """
        Encourages diversity by penalizing similar trajectories.
        Only computes loss for valid timesteps and predicted tracks.

        Args:
            predicted_trajectories: (batch_size, num_agents, num_trajectories, timesteps, 3)
            ground_truth_states_valid: [batch_size, num_agents, num_timesteps] 
            storing validity information for each state.
            tracks_to_predict: [batch_size, num_agents] boolean mask for agents to predict.
        """
        num_trajs = predicted_trajectories.size(dim=-3)
        diversity_loss = 0.0

        if num_trajs < 2:
            return diversity_loss

        for i in range(num_trajs):
            for j in range(i + 1, num_trajs):
                # Compute pairwise distances
                # [batch_size, num_agents, num_timesteps]
                pairwise_dist = torch.norm(
                    predicted_trajectories[..., i, :, :2] -
                    predicted_trajectories[..., j, :, :2], dim=-1)
                
                # Mask out invalid timesteps and non-predicted tracks
                # [batch_size, num_agents, num_timesteps]
                masked_dist = pairwise_dist * ground_truth_states_valid
                # [batch_size, num_agents, 1]
                agent_mask = tracks_to_predict.bool().unsqueeze(-1)
                masked_dist = masked_dist * agent_mask
                
                # Compute mean only over valid timesteps and agents
                valid_count = (ground_truth_states_valid * agent_mask).sum()
                if valid_count > 0:
                    mean_dist = masked_dist.sum() / valid_count
                    # Penalize trajectories that are too close
                    diversity_loss += torch.exp(-mean_dist)

        if num_trajs >= 2:
            diversity_loss /= (num_trajs * (num_trajs - 1)) / 2  # Normalize over number of pairs
        
        return diversity_loss
