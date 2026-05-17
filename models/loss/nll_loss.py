import torch
import torch.nn as nn


# Diversity loss is bounded [0, 1] and decays fast (exp(-mean_dist)), while
# min_ade contributes 5-10 in raw meters. Scale diversity up so it's comparable
# to min_ade when modes are collapsed, encouraging differentiation.
DIVERSITY_WEIGHT = 5.0


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
            DIVERSITY_WEIGHT * self.diversity_Loss(predicted_trajectories,
                               ground_truth_states_valid,
                               tracks_to_predict) +
            self.yaw_loss(predicted_trajectories,
                          ground_truth_trajectory,
                          ade_per_mode,
                          ground_truth_states_valid,
                          tracks_to_predict)
        )

    def _compute_ade_per_trajectory(self, predicted_trajectories,
                                    ground_truth_trajectory,
                                    ground_truth_states_valid):
        """
        Computes the Average Displacement Error (ADE) per trajectory mode over x, y only.

        Args:
            predicted_trajectories: [batch_size, num_agents, num_trajectories, num_timesteps, 4]
            ground_truth_trajectory: [batch_size, num_agents, num_timesteps, 4]
            ground_truth_states_valid: [batch_size, num_agents, num_timesteps]

        Returns:
            Tensor: ADE per trajectory mode [batch_size, num_agents, num_trajectories].
        """
        # x, y displacement only — keeps ADE in meters
        # [batch_size, num_agents, num_trajectories, num_timesteps, 2]
        xy_diff = (predicted_trajectories[..., :2] -
                   ground_truth_trajectory[..., :2].unsqueeze(dim=-3))

        masked_diff = xy_diff * ground_truth_states_valid.unsqueeze(dim=-1).unsqueeze(dim=-3)
        ade_per_mode = torch.norm(masked_diff, dim=-1).sum(dim=-1) / \
            (ground_truth_states_valid.sum(dim=-1).unsqueeze(dim=-1) + 1e-8)

        return ade_per_mode

    def weighted_nll_loss(self, ade_per_mode,
                          predicted_probabilities,
                          ground_truth_states_valid,
                          tracks_to_predict):
        """
        Winner-takes-all NLL: penalize -log(p) for the mode with lowest ADE.

        Args:
            ade_per_mode: [batch_size, num_agents, num_trajectories].
            predicted_probabilities: [batch_size, num_agents, num_trajectories].
            ground_truth_states_valid: [batch_size, num_agents, num_timesteps].
            tracks_to_predict: [batch_size, num_agents] boolean mask for agents to predict.

        Returns:
            Total loss (scalar).
        """
        # [batch_size, num_agents]
        best_mode = torch.argmin(ade_per_mode, dim=-1)
        best_prob = predicted_probabilities.gather(
            dim=-1, index=best_mode.unsqueeze(-1)
        ).squeeze(-1)

        nll = -torch.log(best_prob + 1e-8)

        valid_agents_mask = (ground_truth_states_valid.sum(dim=-1) > 0) & tracks_to_predict.bool()

        if valid_agents_mask.sum() > 0:
            return nll[valid_agents_mask].mean()
        else:
            return torch.tensor(0.0, device=nll.device)

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

    def yaw_loss(self, predicted_trajectories, ground_truth_trajectory,
                 ade_per_mode, ground_truth_states_valid, tracks_to_predict):
        """
        Winner-takes-all MSE on sin/cos yaw channels for the best predicted mode.

        Args:
            predicted_trajectories: [batch_size, num_agents, num_trajectories, num_timesteps, 4]
            ground_truth_trajectory: [batch_size, num_agents, num_timesteps, 4]
            ade_per_mode: [batch_size, num_agents, num_trajectories]
            ground_truth_states_valid: [batch_size, num_agents, num_timesteps]
            tracks_to_predict: [batch_size, num_agents]

        Returns:
            Scalar yaw loss.
        """
        # Pick the best mode per agent
        # [batch_size, num_agents]
        best_mode = torch.argmin(ade_per_mode, dim=-1)

        # Gather the best mode's trajectory: [batch_size, num_agents, num_timesteps, 4]
        best_mode_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        best_mode_idx = best_mode_idx.expand(-1, -1, -1, predicted_trajectories.size(-2),
                                             predicted_trajectories.size(-1))
        best_traj = predicted_trajectories.gather(dim=2, index=best_mode_idx).squeeze(2)

        # MSE on sin and cos channels (indices 2 and 3)
        # [batch_size, num_agents, num_timesteps]
        sin_err = (best_traj[..., 2] - ground_truth_trajectory[..., 2]) ** 2
        cos_err = (best_traj[..., 3] - ground_truth_trajectory[..., 3]) ** 2
        yaw_err = (sin_err + cos_err) * ground_truth_states_valid

        valid_agents_mask = (ground_truth_states_valid.sum(dim=-1) > 0) & tracks_to_predict.bool()

        if valid_agents_mask.sum() == 0:
            return torch.tensor(0.0, device=predicted_trajectories.device)

        valid_counts = ground_truth_states_valid[valid_agents_mask].sum(dim=-1) + 1e-8
        per_agent_yaw_loss = yaw_err[valid_agents_mask].sum(dim=-1) / valid_counts
        return per_agent_yaw_loss.mean()
