import torch
import torch.nn as nn

from models.transformer.polyline_pointnet import PolylinePointNet
from models.transformer.point_encoder import PointEncoder
from models.transformer.categorical_embedder import CategoricalEmbedder
from models.transformer.context_encoder import ContextEncoder
from models.transformer.positional_encoder import PositionalEncoder
from models.transformer.prediction_decoder import PredictionDecoder
from utils.model.endpoint_anchors import (
    ANCHOR_BALLISTIC,
    VALID_ANCHOR_TYPES,
    get_mode_anchor_endpoints,
)


class Transformer_NN(nn.Module):
    """Multi-modal motion prediction transformer.

    Encoder: 2-layer factored attention over (agents, time) + cross-attention
    to static / dynamic roadgraph.
    Decoder: 2-layer agent-axis self-attention + cross-attention to encoded
    past. K trajectory modes are independent in the decoder; they interact
    only through the loss.

    Each decoder query is built as:
        query[b, a, k] = endpoint_encoder(anchor[b, a, k])  +  mode_queries[k]
    where `anchor` is produced by `get_mode_anchor_endpoints(...)` according
    to `anchor_type`. The per-mode `mode_queries` Embedding is initialized
    small so the anchor (which carries semantic mode identity) dominates the
    query early in training.
    """

    def __init__(self, num_agent_features,
                 num_static_road_features,
                 num_dynamic_road_features,
                 num_past_timesteps,
                 num_model_features,
                 categorical_embedding_dim,
                 num_future_trajectories,
                 num_future_timesteps,
                 num_future_features,
                 anchor_type=ANCHOR_BALLISTIC,
                 centroids=None):
        """
        Args:
            anchor_type: one of utils.model.endpoint_anchors.VALID_ANCHOR_TYPES.
                BALLISTIC needs no extra data; CENTROID and HYBRID require
                pre-computed `centroids`.
            centroids: [K_centroids, 2] tensor in agent-local frame, or None.
                Stored as a non-trainable buffer when provided.
        """
        super().__init__()

        if anchor_type not in VALID_ANCHOR_TYPES:
            raise ValueError(
                f"Unknown anchor_type={anchor_type!r}, "
                f"expected one of {VALID_ANCHOR_TYPES}.")

        self.num_past_timesteps      = num_past_timesteps
        self.num_future_trajectories = num_future_trajectories
        self.num_future_timesteps    = num_future_timesteps
        self.num_future_features     = num_future_features
        self.d_model                 = num_model_features
        self.categorical_embedding_dim = categorical_embedding_dim
        self.anchor_type             = anchor_type

        # Centroids: registered as a buffer (saved with state_dict, moves to
        # GPU with .to(device), but never updated by gradient).
        # Empty tensor when not provided keeps state_dict shape consistent.
        if centroids is None:
            centroids = torch.empty(0, 2)
        self.register_buffer('centroids', centroids)

        # --- Context encoder inputs ---------------------------------------
        # Static road polylines: pointnet collapses points -> per-polyline emb.
        self.static_rg_pointnet = PolylinePointNet(
            num_features_per_point=num_static_road_features,
            d_model=self.d_model,
        )
        # Agents and dynamic road: per-point linear encoder leaving room to
        # concat the categorical embedding (agent type / traffic-light state).
        self.past_agent_point_encoder = PointEncoder(
            num_features_per_point=num_agent_features,
            d_model=self.d_model - self.categorical_embedding_dim,
        )
        self.dynamic_rg_point_encoder = PointEncoder(
            num_features_per_point=num_dynamic_road_features,
            d_model=self.d_model - self.categorical_embedding_dim,
        )
        self.categorical_embedder = CategoricalEmbedder(
            embedding_dim=self.categorical_embedding_dim,
        )
        self.context_positional_encoding = PositionalEncoder(
            d_model=self.d_model,
            num_timesteps=num_past_timesteps,
        )

        # --- Transformer blocks --------------------------------------------
        self.transformer_encoder = ContextEncoder(
            num_layers=2, d_model=num_model_features, ffn_hidden=2048,
            num_heads=4, drop_prob=0.1,
        )
        self.transformer_decoder = PredictionDecoder(
            num_layers=2, d_model=num_model_features, ffn_hidden=2048,
            num_heads=4, drop_prob=0.1,
        )

        # --- Decoder query construction ------------------------------------
        # 2D endpoint anchor (in SDC frame) -> d_model. Same encoder for all
        # anchor types; the difference is in how the 2D anchors were produced.
        self.endpoint_encoder = nn.Linear(2, num_model_features)
        # Per-mode learnable refinement on top of the anchor. Small init
        # (std=0.01) so the anchor dominates the query early in training.
        self.mode_queries = nn.Embedding(num_future_trajectories,
                                         num_model_features)
        nn.init.normal_(self.mode_queries.weight, std=0.01)

        # --- Prediction heads ---------------------------------------------
        self.traj_head = nn.Sequential(
            nn.Linear(num_model_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_future_timesteps * num_future_features),
        )
        self.prob_head = nn.Linear(num_model_features, 1)

    def forward(self, agents_continuous, agents_categorical, agents_valid,
                static_road, static_road_valid,
                dynamic_road_continuous, dynamic_road_categorical, dynamic_road_valid,
                capture=False):
        batch_size = agents_continuous.size(0)
        num_agents = agents_continuous.size(1)
        num_dyn_rg = dynamic_road_continuous.size(1)

        # ------- Per-point embeddings (continuous + categorical) -----------
        # Flatten time into the points axis so the per-point encoder sees a
        # single big sequence, then unflatten back to per-timestep tokens.
        agents_flat = agents_continuous.reshape(
            batch_size, num_agents * self.num_past_timesteps, -1)
        agents_cont_emb = self.past_agent_point_encoder(agents_flat).reshape(
            batch_size, num_agents, self.num_past_timesteps, -1)
        agents_cat_emb = self.categorical_embedder(
            agent_type=agents_categorical)['agent_type']
        agent_embeddings = torch.cat([agents_cont_emb, agents_cat_emb], dim=-1)

        dyn_flat = dynamic_road_continuous.reshape(
            batch_size, num_dyn_rg * self.num_past_timesteps, -1)
        dyn_cont_emb = self.dynamic_rg_point_encoder(dyn_flat).reshape(
            batch_size, num_dyn_rg, self.num_past_timesteps, -1)
        dyn_cat_emb = self.categorical_embedder(
            traffic_light_state=dynamic_road_categorical)['traffic_light']
        dynamic_rg_embedding = torch.cat([dyn_cont_emb, dyn_cat_emb], dim=-1)

        # Static map is time-invariant: encode once, broadcast across T.
        static_rg_embedding = self.static_rg_pointnet(static_road).unsqueeze(
            dim=-2,
        ).repeat(1, 1, self.num_past_timesteps, 1)

        # Positional encoding on the time-varying inputs only.
        agent_embeddings     = self.context_positional_encoding(agent_embeddings)
        dynamic_rg_embedding = self.context_positional_encoding(dynamic_rg_embedding)

        # ------- Transformer encoder -------------------------------------
        context_encoded_agents = self.transformer_encoder(
            agent_embeddings, agents_valid,
            static_rg_embedding, static_road_valid,
            dynamic_rg_embedding, dynamic_road_valid,
            capture=capture,
        )

        # ------- Decoder queries from 2D endpoint anchors ----------------
        # Anchor source is determined by self.anchor_type; see
        # utils/model/endpoint_anchors.py for the three implementations.
        endpoints = get_mode_anchor_endpoints(
            self.anchor_type,
            agents_continuous,
            self.num_future_timesteps,
            self.num_future_trajectories,
            centroids=self._centroids_or_none(),
        )  # [B, A, K, 2] in SDC frame
        anchor_emb = self.endpoint_encoder(endpoints)  # [B, A, K, d_model]
        future_agents = (
            anchor_emb
            + self.mode_queries.weight.view(
                1, 1, self.num_future_trajectories, -1)
        )
        future_agents_valid = torch.ones(
            batch_size, num_agents, self.num_future_trajectories,
            device=agents_continuous.device, dtype=agents_continuous.dtype,
        )

        # ------- Transformer decoder -------------------------------------
        decoded_agents = self.transformer_decoder(
            context_encoded_agents, agents_valid,
            future_agents, future_agents_valid,
            capture=capture,
        )

        # ------- Prediction heads ----------------------------------------
        future_trajectories = self.traj_head(decoded_agents).view(
            decoded_agents.size(0), decoded_agents.size(1),
            self.num_future_trajectories,
            self.num_future_timesteps, self.num_future_features,
        )
        probs = self.prob_head(decoded_agents).view(
            decoded_agents.size(0), decoded_agents.size(1),
            self.num_future_trajectories,
        )
        probs = torch.softmax(probs, dim=-1)

        return future_trajectories, probs

    def _centroids_or_none(self):
        """Return self.centroids unless it's the empty placeholder buffer."""
        return self.centroids if self.centroids.numel() > 0 else None
