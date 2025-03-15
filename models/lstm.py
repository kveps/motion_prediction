import torch
import torch.nn as nn


class LSTM_NN(nn.Module):
    def __init__(self, num_agent_features,
                 num_static_road_features,
                 num_dynamic_road_features,
                 agent_hidden_size,
                 static_roadgraph_hidden_size,
                 dynamic_roadgraph_hidden_size,
                 num_future_features,
                 num_future_timesteps):
        super(LSTM_NN, self).__init__()

        self.future_timesteps = num_future_timesteps

        # Agent LSTM layer
        self.agent_lstm = nn.LSTM(
            num_agent_features, agent_hidden_size, batch_first=True)

        # Static Roadgraph layer
        self.static_road_fc = nn.Linear(
            num_static_road_features, static_roadgraph_hidden_size)
        self.static_relu = nn.ReLU()

        # Dynamic Roadgraph layer
        self.dynamic_road_lstm = nn.LSTM(
            num_dynamic_road_features, dynamic_roadgraph_hidden_size,
            batch_first=True)

        # Combine all inputs
        self.combined_fc = nn.Linear(
            agent_hidden_size +
            static_roadgraph_hidden_size +
            dynamic_roadgraph_hidden_size, agent_hidden_size)
        self.combined_relu = nn.ReLU()

        # Future Timestep Prediction
        self.future_fc = nn.Linear(
            agent_hidden_size, num_future_features * num_future_timesteps)

    def forward(self, agents, static_road, dynamic_road):
        batch_size, num_agents, num_timesteps, _ = agents.size()
        _, num_static_roadgraph_samples, _ = static_road.size()
        _, num_tl_states, num_timesteps_dynamic, _ = dynamic_road.size()

        # Agent Processing
        # [batch_size*num_agents, num_timesteps, agent_features]
        agents_reshaped = agents.view(
            batch_size * num_agents, num_timesteps, -1)  # Reshape for LSTM
        # [batch_size*num_agents, num_timesteps, agent_hidden_size]
        agent_lstm_out, _ = self.agent_lstm(agents_reshaped)
        # Take the last timestep
        # [batch_size*num_agents, agent_hidden_size]
        agent_last_timestep = agent_lstm_out[:, -1, :]
        # Reshape back to [batch, num_agents, agent_hidden_size]
        agent_out = agent_last_timestep.view(batch_size, num_agents, -1)

        # Static Roadgraph Processing
        # [batch_size, num_static_roadgraph_samples, static_roadgraph_hidden_size]
        static_road_out = self.static_road_fc(static_road)
        # [batch_size, num_static_roadgraph_samples, static_roadgraph_hidden_size]
        static_road_out = self.static_relu(static_road_out)
        # average roadgraph samples, and repeat for each agent [batch, num_agents, road_hidden]
        static_road_out = torch.mean(static_road_out, dim=1).unsqueeze(
            1).repeat(1, num_agents, 1)

        # Dynamic Roadgraph Processing
        # [batch_size*num_tl_states, num_timesteps_dynamic, dynamic_roadgraph_features]
        dynamic_road_reshaped = dynamic_road.view(
            batch_size * num_tl_states, num_timesteps_dynamic, -1)  # Reshape for LSTM
        # [batch_size*num_tl_states, num_timesteps_dynamic, dynamic_roadgraph_hidden_size]
        dynamic_road_lstm_out, _ = self.dynamic_road_lstm(
            dynamic_road_reshaped)
        # [batch_size*num_tl_states, dynamic_roadgraph_hidden_size]
        dynamic_road_last_timestep = dynamic_road_lstm_out[:, -1, :]
        # [batch_size, num_tl_states, dynamic_roadgraph_hidden_size]
        dynamic_road_out = dynamic_road_last_timestep.view(
            batch_size, num_tl_states, -1)
        # average roadgraph samples, and repeat for each agent [batch, num_agents, dynamic_roadgraph_hidden_size]
        dynamic_road_out = torch.mean(dynamic_road_out, dim=1).unsqueeze(
            1).repeat(1, num_agents, 1)

        # Combine and Output
        # concatenate along feature dimension.
        # [batch, num_agents, combined_hidden_size]
        combined = torch.cat(
            (agent_out, static_road_out, dynamic_road_out), dim=-1)
        combined = self.combined_fc(combined)
        combined = self.combined_relu(combined)

        # Future Timestep Prediction
        # [batch_size, num_agents, num_future_features * num_future_timesteps]
        output = self.future_fc(combined)

        # [batch_size, num_agents, num_future_timesteps, num_future_features]
        output = output.view(batch_size, num_agents, self.future_timesteps,
                             -1)

        return output
