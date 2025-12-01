import torch
import torch.nn as nn
from utils.data.features_description import (
    NUM_TL_CATEGORIES,
    NUM_AGENT_CATEGORIES,
)

class CategoricalEmbedder(nn.Module):
    """
    Handles embedding of categorical features based on Waymo Open Motion Dataset (WOMD).
    """
    
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Traffic light state embedding (9 states)
        self.traffic_light_embedding = nn.Embedding(num_embeddings=NUM_TL_CATEGORIES, embedding_dim=embedding_dim)
        
        # Agent type embedding (4 types)
        self.agent_type_embedding = nn.Embedding(num_embeddings=NUM_AGENT_CATEGORIES, embedding_dim=embedding_dim)
    
    def forward(self, traffic_light_state=None, agent_type=None):
        """
        Args:
            traffic_light_state: Tensor of shape [...], integer indices 0-8
            agent_type: Tensor of shape [...], integer indices 0-4
        
        Returns:
            Dict with embedded tensors
        """
        embeddings = {}
        if traffic_light_state is not None:
            embeddings['traffic_light'] = self.traffic_light_embedding(traffic_light_state.long())
        if agent_type is not None:
            embeddings['agent_type'] = self.agent_type_embedding(agent_type.long())
        return embeddings
