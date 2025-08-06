import torch
import torch.nn as nn
import torch.futures

# Neural Network For Implied Volatility
class IV(nn.Module):
    def __init__(
        self, input_features=21, hidden_size=200, dropout_rate=0.0, num_hidden_layers=6
    ):
        super().__init__()
        # Activation function
        activation = nn.Mish()
        # Layer normalization
        ln = nn.LayerNorm(hidden_size)
        # Create list of layers
        layers = []

        # Input layer
        layers.extend(
            [
                nn.Linear(input_features, hidden_size),
                ln,
                activation,
            ]
        )

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    ln,
                    activation,
                    nn.Dropout(dropout_rate),
                ]
            )

        # Combine all hidden layers into a Sequential
        self.hidden_layers = nn.Sequential(*layers)
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = torch.nn.functional.softplus(self.output_layer(x))
        return x

class Joint(nn.Module):
    def __init__(
        self, input_features=7, hidden_size=200, dropout_rate=0.0, num_hidden_layers=6
    ):
        super().__init__()
        # Activation function
        activation = nn.Mish()
        # Layer normalization
        ln = nn.LayerNorm(hidden_size)
        # Create list of layers
        layers = []

        # Input layer
        layers.extend(
            [
                nn.Linear(input_features, hidden_size),
                ln,
                activation,
            ]
        )

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    ln,
                    activation,
                    nn.Dropout(dropout_rate),
                ]
            )

        # Combine all hidden layers into a Sequential
        self.hidden_layers = nn.Sequential(*layers)
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 5)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = torch.nn.functional.softplus(self.output_layer(x))
        return x
