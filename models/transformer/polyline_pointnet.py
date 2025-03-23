import torch
import torch.nn as nn


class PolylinePointNet(nn.Module):
    def __init__(self, num_features_per_point, d_model, embedding_size=64):
        super(PolylinePointNet, self).__init__()
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

    def forward(self, polylines):
        """
        Encodes a batch of polylines using PointNet.

        Args:
            polylines (torch.Tensor):
            Tensor of shape (batch_size,
                            num_objects,
                            num_points_per_polyline,
                            num_features_per_point).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_objects,d_model), 
            representing the encoded polylines.
        """
        batch_size, num_objects, num_points, num_features = polylines.size()

        # Feature Extraction (MLP1)
        x = polylines.view(-1, num_features)  # Reshape for MLP
        x = self.mlp1(x)
        # Reshape back
        x = x.view(batch_size*num_objects, num_points, self.embedding_size)

        # Feature Aggregation (Max Pooling)
        # [batch_size*num_objects, embedding_size]
        x = torch.max(x, dim=-2, keepdim=False)[0]  # Max pooling over points

        # Global Feature Extraction (MLP2)
        # [batch_size*num_objects, d_model]
        x = self.mlp2(x)
        x = x.view(batch_size, num_objects, self.d_model)

        return x
