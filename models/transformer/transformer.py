import torch
import torch.nn as nn
from models.transformer.polyline_pointnet import PolylinePointNet
from models.transformer.point_encoder import PointEncoder
from models.transformer.categorical_embedder import CategoricalEmbedder
from models.transformer.context_encoder import ContextEncoder
from models.transformer.positional_encoder import PositionalEncoder
from models.transformer.prediction_decoder import PredictionDecoder


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

        # Point encoding for future agents (single point encoding, no aggregation)
        self.future_agent_point_encoder = PointEncoder(
            num_features_per_point=num_future_features,
            d_model=self.d_model)

        # Positional embedding for future
        self.future_positional_encoding = PositionalEncoder(
            d_model=self.d_model,
            num_timesteps=self.num_future_timesteps,
        )

        # Transformer encoder
        self.transformer_encoder = ContextEncoder(
            num_layers=2, d_model=num_model_features, ffn_hidden=2048, num_heads=4, drop_prob=0.1,
        )

        # Transformer decoders (for each future trajectory)
        self.transformer_decoders = nn.ModuleList([PredictionDecoder(
            num_layers=2, d_model=num_model_features, ffn_hidden=2048, num_heads=4, drop_prob=0.1,
        ) for _ in range(num_future_trajectories)])

        # Trajectory Prediction
        self.trajectory_predictions = nn.ModuleList([nn.Linear(
            num_model_features, num_future_features) for _ in range(num_future_trajectories)])

        # Trajectory probabilities
        self.trajectory_confidences = nn.ModuleList([
            nn.Linear(num_model_features*num_future_timesteps, 1)
            for _ in range(num_future_trajectories)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, agents_continuous, agents_categorical, agents_valid, 
                static_road, static_road_valid, 
                dynamic_road_continuous, dynamic_road_categorical, dynamic_road_valid, 
                future_agents, future_agents_valid):
        # Encode past context
        agent_embeddings = torch.zeros(
            (agents_continuous.size(0), agents_continuous.size(1), agents_continuous.size(2), self.d_model),
            device=agents_continuous.device, dtype=agents_continuous.dtype)
        dynamic_rg_embedding = torch.zeros(
            (dynamic_road_continuous.size(0), dynamic_road_continuous.size(1), dynamic_road_continuous.size(2), self.d_model),
            device=dynamic_road_continuous.device, dtype=dynamic_road_continuous.dtype)
        
        for t in range(self.num_past_timesteps):
            # Agent continuous encoding
            agents_cont_t = agents_continuous[..., t, :]  
            agents_cont_emb = self.past_agent_point_encoder(agents_cont_t) 
            
            # Agent categorical embedding
            agents_cat_t = agents_categorical[..., t]  
            agents_cat_emb = self.categorical_embedder(agent_type=agents_cat_t)['agent_type'] 
            
            # Concatenate
            agent_embeddings[..., t, :] = torch.cat([agents_cont_emb, agents_cat_emb], dim=-1)
            
            # Dynamic road continuous encoding
            dyn_cont_t = dynamic_road_continuous[..., t, :]  
            dyn_cont_emb = self.dynamic_rg_point_encoder(dyn_cont_t)  
            
            # Dynamic road categorical embedding
            dyn_cat_t = dynamic_road_categorical[..., t]  
            dyn_cat_emb = self.categorical_embedder(traffic_light_state=dyn_cat_t)['traffic_light']  
            
            # Concatenate
            dynamic_rg_embedding[..., t, :] = torch.cat([dyn_cont_emb, dyn_cat_emb], dim=-1)
        
        # Static road polyline embedding
        static_rg_embedding = self.static_rg_pointnet(static_road).unsqueeze(
            dim=-2
        ).repeat(1, 1, self.num_past_timesteps, 1)

        # Positional encoding
        agent_embeddings = self.context_positional_encoding(agent_embeddings)
        dynamic_rg_embedding = self.context_positional_encoding(dynamic_rg_embedding)
        static_rg_embedding = self.context_positional_encoding(static_rg_embedding)

        # Transformer encoder
        context_encoded_agents = self.transformer_encoder(
            agent_embeddings,
            agents_valid,
            static_rg_embedding,
            static_road_valid,
            dynamic_rg_embedding,
            dynamic_road_valid
        )

        # Point encoding for future agents (no sequence aggregation needed)
        future_agent_embeddings = torch.zeros(
            (future_agents.size(0), future_agents.size(
                1), future_agents.size(2), self.d_model),
            device=future_agents.device, dtype=future_agents.dtype
        )
        for t in range(self.num_future_timesteps):
            future_agents_at_t = future_agents[..., t, :]  # [batch_size, num_agents, num_features]
            future_agent_embeddings[..., t, :] = self.future_agent_point_encoder(future_agents_at_t)

        # Positional encoding
        future_agent_embeddings = self.future_positional_encoding(
            future_agent_embeddings
        )

        # Transformer decoder
        future_trajectories = torch.zeros_like(future_agents).unsqueeze(dim=-3).expand(
            -1, -1, self.num_future_trajectories, -1, -1
        )
        confidence_scores = torch.zeros(
            (future_agents.size(0), future_agents.size(1),
             self.num_future_trajectories, 1),
            device=future_agents.device, dtype=future_agents.dtype
        )

        for i in range(self.num_future_trajectories):
            decoder_outputs = self.transformer_decoders[i](
                context_encoded_agents,
                agents_valid,
                # TODO: Initialize each decoder with a separate future agent embedding
                future_agent_embeddings,
                future_agents_valid,
            )
            # Trajectory prediction
            future_trajectories[..., i, :, :] = self.trajectory_predictions[i](
                decoder_outputs)
            # Trajectory confidence
            decoder_outputs = decoder_outputs.view(
                decoder_outputs.size(0), decoder_outputs.size(1),
                self.num_future_timesteps*self.d_model)
            confidence_scores[..., i, :] = self.sigmoid(
                self.trajectory_confidences[i](decoder_outputs))

        # Softmax on confidence scores for trajectory probabilities
        probs = torch.softmax(confidence_scores, dim=-1).squeeze(dim=-1)

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
    
    future_agents = torch.randn(batch_size, num_agents,
                                num_future_timesteps, num_future_features)
    future_agents_valid = torch.ones(
        batch_size, num_agents, num_future_timesteps)

    output = model(agents_continuous, agents_categorical, agents_valid, 
                   static_road, static_road_valid,
                   dynamic_road_continuous, dynamic_road_categorical, dynamic_road_valid, 
                   future_agents, future_agents_valid)
    print("Successful forward pass through Transformer_NN model.")
