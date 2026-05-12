import torch
import torch.nn as nn
from models.transformer.polyline_pointnet import PolylinePointNet
from models.transformer.point_encoder import PointEncoder
from models.transformer.categorical_embedder import CategoricalEmbedder
from models.transformer.context_encoder import ContextEncoder
from models.transformer.positional_encoder import PositionalEncoder
from models.transformer.prediction_decoder import PredictionDecoder
from utils.model.engineered_predictions import ballistic_trajectories


class Transformer_NN(nn.Module):
    def __init__(self, num_agent_features,
                 num_static_road_features,
                 num_dynamic_road_features,
                 num_past_timesteps,
                 num_model_features,
                 categorical_embedding_dim,
                 num_future_trajectories,
                 num_future_timesteps,
                 num_future_features):
        super(Transformer_NN, self).__init__()

        self.num_past_timesteps = num_past_timesteps
        self.num_future_trajectories = num_future_trajectories
        self.num_future_timesteps = num_future_timesteps
        self.num_future_features = num_future_features
        self.d_model = num_model_features
        self.categorical_embedding_dim = categorical_embedding_dim

        # Polyline embedding for static road (true polyline with multiple points)
        self.static_rg_pointnet = PolylinePointNet(
            num_features_per_point=num_static_road_features,
            d_model=self.d_model)
        
        # Point encoding for agents and dynamic road (continuous features only)
        # Categorical features will be embedded separately and concatenated
        self.past_agent_point_encoder = PointEncoder(
            num_features_per_point=num_agent_features,
            d_model=self.d_model - self.categorical_embedding_dim)  
        self.dynamic_rg_point_encoder = PointEncoder(
            num_features_per_point=num_dynamic_road_features,
            d_model=self.d_model - self.categorical_embedding_dim)
        
        # Categorical embeddings
        self.categorical_embedder = CategoricalEmbedder(embedding_dim=self.categorical_embedding_dim)

        # Positional embedding for all inputs
        self.context_positional_encoding = PositionalEncoder(
            d_model=self.d_model,
            num_timesteps=num_past_timesteps,
        )

        # Transformer encoder
        self.transformer_encoder = ContextEncoder(
            num_layers=2, d_model=num_model_features, ffn_hidden=2048, num_heads=4, drop_prob=0.1,
        )

        # Transformer decoder
        self.transformer_decoder = PredictionDecoder(
            num_layers=2, d_model=num_model_features, ffn_hidden=2048, num_heads=4, drop_prob=0.1,
        )

        # Ballistic endpoint encoder: projects (x_end, y_end) → d_model
        self.ballistic_encoder = nn.Linear(2, num_model_features)
        # Per-mode offsets on top of the ballistic base for multi-modal diversity
        self.mode_queries = nn.Embedding(num_future_trajectories, num_model_features)

        # Trajectory Head
        self.traj_head = nn.Sequential(
            nn.Linear(num_model_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_future_timesteps * num_future_features)
        )

        # Probability Head
        self.prob_head = nn.Linear(num_model_features, 1)

    def forward(self, agents_continuous, agents_categorical, agents_valid,
                static_road, static_road_valid,
                dynamic_road_continuous, dynamic_road_categorical, dynamic_road_valid):
        batch_size = agents_continuous.size(0)
        num_agents = agents_continuous.size(1)
        num_dyn_rg = dynamic_road_continuous.size(1)

        # Encode all past timesteps at once by merging the time axis into the objects axis
        # [batch, agents, timesteps, features] → [batch, agents*timesteps, features]
        agents_flat = agents_continuous.reshape(
            batch_size, num_agents * self.num_past_timesteps, -1)
        agents_cont_emb = self.past_agent_point_encoder(agents_flat).reshape(
            batch_size, num_agents, self.num_past_timesteps, -1)
        agents_cat_emb = self.categorical_embedder(
            agent_type=agents_categorical)['agent_type']
        agent_embeddings = torch.cat([agents_cont_emb, agents_cat_emb], dim=-1)

        # Same for dynamic road
        dyn_flat = dynamic_road_continuous.reshape(
            batch_size, num_dyn_rg * self.num_past_timesteps, -1)
        dyn_cont_emb = self.dynamic_rg_point_encoder(dyn_flat).reshape(
            batch_size, num_dyn_rg, self.num_past_timesteps, -1)
        dyn_cat_emb = self.categorical_embedder(
            traffic_light_state=dynamic_road_categorical)['traffic_light']
        dynamic_rg_embedding = torch.cat([dyn_cont_emb, dyn_cat_emb], dim=-1)
        
        # Static road polyline embedding
        static_rg_embedding = self.static_rg_pointnet(static_road).unsqueeze(
            dim=-2
        ).repeat(1, 1, self.num_past_timesteps, 1)

        # Positional encoding — static road is time-invariant, no PE applied
        agent_embeddings = self.context_positional_encoding(agent_embeddings)
        dynamic_rg_embedding = self.context_positional_encoding(dynamic_rg_embedding)

        # Transformer encoder
        context_encoded_agents = self.transformer_encoder(
            agent_embeddings,
            agents_valid,
            static_rg_embedding,
            static_road_valid,
            dynamic_rg_embedding,
            dynamic_road_valid
        )

        # Build decoder queries from ballistic prediction + per-mode learned offsets
        # [batch, agents, T, 4] → endpoint [batch, agents, 2]
        ballistic_endpoint = ballistic_trajectories(
            agents_continuous, self.num_future_timesteps)[..., -1, :2]
        ballistic_query = self.ballistic_encoder(ballistic_endpoint)
        # [batch, agents, d_model] + [K, d_model] → [batch, agents, K, d_model]
        future_agents = (ballistic_query.unsqueeze(2) +
                         self.mode_queries.weight.unsqueeze(0).unsqueeze(0))
        future_agents_valid = torch.ones(
            batch_size, num_agents, self.num_future_trajectories,
            device=agents_continuous.device, dtype=agents_continuous.dtype)

        # Transformer decoder
        decoded_agents = self.transformer_decoder(
            context_encoded_agents,
            agents_valid,
            future_agents,
            future_agents_valid,
        )

        # Prediction heads
        #
        # Trajectory head
        future_trajectories = self.traj_head(decoded_agents)
        future_trajectories = future_trajectories.view(
            decoded_agents.size(0), decoded_agents.size(1),
            self.num_future_trajectories, self.num_future_timesteps, self.num_future_features
        )
        # Probability head
        probs = self.prob_head(decoded_agents)
        probs = probs.view(
            decoded_agents.size(0), decoded_agents.size(1),
            self.num_future_trajectories   
        )
        probs = torch.softmax(probs, dim=-1)

        return future_trajectories, probs


# Example usage
test_usage = False
if test_usage:
    num_agent_continuous_features = 8
    num_agent_categorical_features = 1
    num_static_road_features = 4
    num_dynamic_road_continuous_features = 2
    num_dynamic_road_categorical_features = 1
    num_past_timesteps = 11
    num_model_features = 256
    categorical_embedding_dim = 16
    num_future_timesteps = 80
    num_future_features = 4
    num_future_trajectories = 3

    model = Transformer_NN(num_agent_features=num_agent_continuous_features,
                           num_static_road_features=num_static_road_features,
                           num_dynamic_road_features=num_dynamic_road_continuous_features,
                           num_past_timesteps=num_past_timesteps,
                           num_model_features=num_model_features,
                           categorical_embedding_dim=categorical_embedding_dim,
                           num_future_trajectories=num_future_trajectories,
                           num_future_timesteps=num_future_timesteps,
                           num_future_features=num_future_features)

    batch_size = 10
    num_agents = 10
    num_static_rg = 500
    num_static_points_per_polyline = 20
    num_dynamic_rg = 7
    
    # Separate continuous and categorical features
    agents_continuous = torch.randn(batch_size, num_agents,
                                    num_past_timesteps, num_agent_continuous_features)
    agents_categorical = torch.randint(0, 5, (batch_size, num_agents, num_past_timesteps))  # 5 agent types
    agents_valid = torch.ones(batch_size, num_agents, num_past_timesteps)
    
    static_road = torch.randn(batch_size, num_static_rg,
                              num_static_points_per_polyline, num_static_road_features)
    static_road_valid = torch.ones(
        batch_size, num_static_rg, num_static_points_per_polyline)
    
    dynamic_road_continuous = torch.randn(batch_size, num_dynamic_rg,
                                          num_past_timesteps, num_dynamic_road_continuous_features)
    dynamic_road_categorical = torch.randint(0, 9, (batch_size, num_dynamic_rg, num_past_timesteps))  # 9 traffic light states
    dynamic_road_valid = torch.ones(
        batch_size, num_dynamic_rg, num_past_timesteps)
    
    output = model(agents_continuous, agents_categorical, agents_valid,
                   static_road, static_road_valid,
                   dynamic_road_continuous, dynamic_road_categorical, dynamic_road_valid)
    print("Successful forward pass through Transformer_NN model.")
