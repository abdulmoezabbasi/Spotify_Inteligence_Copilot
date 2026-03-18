"""Neural network model for Spotify track popularity prediction."""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import os


class PopularityNet(nn.Module):
    """PyTorch neural network for predicting track popularity.
    
    Architecture: Dense layers with BatchNorm, ReLU, and Dropout.
    Input: 16 audio features
    Output: Popularity score (0-100)
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3):
        """Initialize the neural network.
        
        Args:
            input_dim: Number of input features (typically 16)
            hidden_dims: List of hidden layer dimensions, e.g., [128, 64, 32]
            dropout_rate: Dropout probability for regularization
        """
        super(PopularityNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with BatchNorm, ReLU, and Dropout
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer (single neuron for regression)
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predicted popularity scores of shape (batch_size, 1)
        """
        return self.network(x)


def load_model(model_dir: str, device: str = 'cpu') -> Tuple[PopularityNet, List[str]]:
    """Load a trained popularity model from disk.
    
    Args:
        model_dir: Path to directory containing model_config.pt and popularity_model.pt
        device: Device to load model onto ('cpu' or 'cuda')
        
    Returns:
        Tuple of (model, feature_names) where model is on the specified device
    """
    model_config = torch.load(
        os.path.join(model_dir, 'model_config.pt'),
        weights_only=False,
        map_location=device
    )
    
    model = PopularityNet(
        model_config['input_dim'],
        model_config['hidden_dims'],
        model_config['dropout']
    )
    
    model.load_state_dict(
        torch.load(
            os.path.join(model_dir, 'popularity_model.pt'),
            weights_only=False,
            map_location=device
        )
    )
    
    model.to(device)
    model.eval()
    
    return model, model_config['features']


def predict(model: PopularityNet, features: torch.Tensor, scaler=None) -> float:
    """Make a prediction using the model.
    
    Args:
        model: The PopularityNet model
        features: Scaled feature tensor of shape (1, 16)
        scaler: Optional scaler for additional post-processing
        
    Returns:
        Predicted popularity score (0-100)
    """
    with torch.no_grad():
        prediction = model(features).item()
    
    # Clamp to valid popularity range
    return max(0, min(100, prediction))
