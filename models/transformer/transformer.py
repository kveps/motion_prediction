import torch
import torch.nn as nn
from models.transformer.polyline_pointnet import PolylinePointNet
from models.transformer.context_encoder import ContextEncoder
from models.transformer.positional_encoder import PositionalEncoder
from models.transformer.prediction_decoder import PredictionDecoder


class Transformer_NN(nn.Module):
    def __init__(self, num_agent_features,
                 num_static_road_features,
                 num_dynamic_road_features,
                 num_past_timesteps,
                 num_model_features,
                 num_future_timesteps,
                 num_future_features):
        super(Transformer_NN, self).__init__()

        self.num_past_timesteps = num_past_timesteps
        self.future_timesteps = num_future_timesteps
        self.d_model = num_model_features

        # Polyline embedding for all inputs (agent, static_rg, dynamic_rg)
        self.static_rg_pointnet = PolylinePointNet(
            num_features_per_point=num_static_road_features,
            d_model=self.d_model)
        self.dynamic_rg_pointnet = PolylinePointNet(
            num_features_per_point=num_dynamic_road_features,
            d_model=self.d_model)
        self.past_agent_pointnet = PolylinePointNet(
            num_features_per_point=num_agent_features,
            d_model=self.d_model)

        # Positional embedding for all inputs
        self.context_positional_encoding = PositionalEncoder(
            d_model=self.d_model,
            num_timesteps=num_past_timesteps,
        )

        # Polyline embedding for future
        self.future_agent_pointnet = PolylinePointNet(
            num_features_per_point=num_future_features,
            d_model=self.d_model)

        # Positional embedding for future
        self.future_positional_encoding = PositionalEncoder(
            d_model=self.d_model,
            num_timesteps=self.future_timesteps,
        )

        # Transformer encoder
        self.transformer_encoder = ContextEncoder(
            num_layers=2, d_model=num_model_features, ffn_hidden=2048, num_heads=4, drop_prob=0.1,
        )

        # Transformer decoder
        self.transformer_decoder = PredictionDecoder(
            num_layers=2, d_model=num_model_features, ffn_hidden=2048, num_heads=4, drop_prob=0.1,
        )

        # Trajectory Prediction
        self.linear = nn.Linear(self.d_model, num_future_features)

    def forward(self, agents, agents_valid, static_road, static_road_valid, dynamic_road, dynamic_road_valid, future_agents, future_agents_valid):
        # Polyline embeddings
        #
        agent_embeddings = torch.zeros(
            (agents.size(0), agents.size(1), agents.size(2), self.d_model))
        dynamic_rg_embedding = torch.zeros(
            (dynamic_road.size(0), dynamic_road.size(1), dynamic_road.size(2), self.d_model))
        for t in range(self.num_past_timesteps):
            agents_at_t = agents[..., t, :].unsqueeze(dim=-2)
            dynamic_road_at_t = dynamic_road[..., t, :].unsqueeze(dim=-2)
            agent_embeddings[..., t, :] = self.past_agent_pointnet(agents_at_t)
            dynamic_rg_embedding[..., t, :] = self.dynamic_rg_pointnet(
                dynamic_road_at_t)
        static_rg_embedding = self.static_rg_pointnet(static_road).unsqueeze(
            dim=-2
        ).repeat(1, 1, self.num_past_timesteps, 1)

        # Positional encoding
        agent_embeddings = self.context_positional_encoding(agent_embeddings)
        dynamic_rg_embedding = self.context_positional_encoding(
            dynamic_rg_embedding)
        static_rg_embedding = self.context_positional_encoding(
            static_rg_embedding)

        # Transformer encoder
        context_encoded_agents = self.transformer_encoder(
            agent_embeddings,
            agents_valid,
            static_rg_embedding,
            static_road_valid,
            dynamic_rg_embedding,
            dynamic_road_valid
        )

        # Polyline embeddings for future
        future_agent_embeddings = torch.zeros(
            (future_agents.size(0), future_agents.size(
                1), future_agents.size(2), self.d_model)
        )
        for t in range(self.future_timesteps):
            future_agents_at_t = future_agents[..., t, :].unsqueeze(dim=-2)
            future_agent_embeddings[..., t, :] = self.future_agent_pointnet(
                future_agents_at_t)

        # Positional encoding
        future_agent_embeddings = self.future_positional_encoding(
            future_agent_embeddings
        )

        # Transformer decoder
        decoder_outputs = self.transformer_decoder(
            context_encoded_agents,
            agents_valid,
            future_agent_embeddings,
            future_agents_valid,
        )

        # Trajectory prediction
        future_trajectory = self.linear(decoder_outputs)

        return future_trajectory


# Example usage
num_agent_features = 10
num_static_road_features = 4
num_dynamic_road_features = 4
num_past_timesteps = 11
num_model_features = 256
num_future_timesteps = 80
num_future_features = 4

model = Transformer_NN(num_agent_features=num_agent_features,
                       num_static_road_features=num_static_road_features,
                       num_dynamic_road_features=num_dynamic_road_features,
                       num_past_timesteps=num_past_timesteps,
                       num_model_features=num_model_features,
                       num_future_timesteps=num_future_timesteps,
                       num_future_features=num_future_features)

batch_size = 10
num_agents = 10
num_static_rg = 500
num_static_points_per_polyline = 20
num_dynamic_rg = 7
agents = torch.randn(batch_size, num_agents,
                     num_past_timesteps, num_agent_features)
agents_valid = torch.ones(batch_size, num_agents, num_past_timesteps)
static_road = torch.randn(batch_size, num_static_rg,
                          num_static_points_per_polyline, num_static_road_features)
static_road_valid = torch.ones(
    batch_size, num_static_rg, num_static_points_per_polyline)
dynamic_road = torch.randn(batch_size, num_dynamic_rg,
                           num_past_timesteps, num_dynamic_road_features)
dynamic_road_valid = torch.ones(batch_size, num_dynamic_rg, num_past_timesteps)
future_agents = torch.randn(batch_size, num_agents,
                            num_future_timesteps, num_future_features)
future_agents_valid = torch.ones(batch_size, num_agents, num_future_timesteps)

output = model(agents, agents_valid, static_road, static_road_valid,
               dynamic_road, dynamic_road_valid, future_agents, future_agents_valid)
