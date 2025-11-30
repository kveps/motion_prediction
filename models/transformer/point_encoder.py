import torch
import torch.nn as nn


class PointEncoder(nn.Module):
    """
    Encodes individual points (no sequence aggregation).
    Used for encoding single agents and dynamic road features.
    
    Unlike PolylinePointNet which aggregates sequences of points,
    this class simply applies MLPs to individual points.
    """
    def __init__(self, num_features_per_point, d_model, embedding_size=64):
        super(PointEncoder, self).__init__()
        self.num_features_per_point = num_features_per_point
        self.embedding_size = embedding_size
        self.d_model = d_model

        self.mlp1 = nn.Sequential(
            nn.Linear(num_features_per_point, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

    def forward(self, points):
        """
        Encodes a batch of individual points using MLPs.

        Args:
            points (torch.Tensor):
            Tensor of shape (batch_size, num_objects, num_features_per_point).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_objects, d_model), 
            representing the encoded points.
        """
        batch_size, num_objects, num_features = points.size()

        # Feature Extraction (MLP1)
        x = points.view(-1, num_features)  # Reshape for MLP
        x = self.mlp1(x)

        # Global Feature Extraction (MLP2)
        x = self.mlp2(x)
        x = x.view(batch_size, num_objects, self.d_model)

        return x
