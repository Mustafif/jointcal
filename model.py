import torch
import torch.nn as nn
import torch.futures
import torch.nn.functional as F


class IV_WideThenDeep(nn.Module):
    def __init__(self, input_features=21, dropout_rate=0.0):
        super().__init__()

        activation = nn.Mish()

        # Wide → Medium → Narrow funnel
        layer_sizes = [512, 256, 128, 64, 32, 16]

        layers = []
        in_size = input_features
        for size in layer_sizes:
            layers.extend([
                nn.Linear(in_size, size),
                nn.BatchNorm1d(size),  # faster convergence
                activation,
                nn.Dropout(dropout_rate)
            ])
            in_size = size

        self.feature_extractor = nn.Sequential(*layers)

        # Output head (small 2-layer)
        self.output_head = nn.Sequential(
            nn.Linear(layer_sizes[-1], 8),
            activation,
            nn.Linear(8, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.softplus(self.output_head(x))  # keep positive output
        return x

class GLUBlock(nn.Module):
    """
    Pre-norm residual GLU block:
      x -> LN -> Linear(2H) -> chunk -> a * sigmoid(b) -> Dropout -> + x
    """
    def __init__(self, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 2 * hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.norm(x)
        a, b = self.fc(h).chunk(2, dim=-1)
        gated = a * torch.sigmoid(b)
        return x + self.dropout(gated)  # residual

class IV_GLU(nn.Module):
    """
    IV model using a GLU-based deep MLP (6 residual GLU blocks).
    Keeps Softplus output for positive targets.
    """
    def __init__(
        self,
        input_features: int = 30,
        hidden_size: int = 350,
        dropout_rate: float = 0.0,
        num_hidden_layers: int = 7,   # fixed at 6 per your request (can keep flexible)
        softplus_beta: float = 1.1    # raise for steeper positive constraint if needed
    ):
        super().__init__()

        # Stem: project inputs to hidden size
        self.stem = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )

        # 6 GLU residual blocks
        self.blocks = nn.Sequential(
            *[GLUBlock(hidden_size, dropout_rate) for _ in range(num_hidden_layers)]
        )

        # Head: light nonlinearity + output
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Mish(),
            nn.Linear(hidden_size, 1)
        )

        self.softplus_beta = softplus_beta
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        # Keep positivity if your target is strictly positive
        return F.softplus(x, beta=self.softplus_beta)

# Neural Network For Implied Volatility
class IV(nn.Module):
    def __init__(
        self, input_features=26, hidden_size=200, dropout_rate=0.0, num_hidden_layers=6
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
        x = self.output_layer(x)

        # Apply constraints to ensure valid GARCH parameters
        # Split output into individual parameters
        alpha = torch.sigmoid(x[:, 0:1]) * 0.5  # alpha in (0, 0.5)
        beta = torch.sigmoid(x[:, 1:2]) * 0.9   # beta in (0, 0.9)
        omega = torch.nn.functional.softplus(x[:, 2:3]) * 0.01  # omega > 0, small
        gamma = torch.tanh(x[:, 3:4]) * 2.0     # gamma in (-2, 2)
        lambda_ = torch.tanh(x[:, 4:5]) * 0.5   # lambda in (-0.5, 0.5)

        # Ensure stationarity: alpha + beta < 1
        # Adjust beta if needed
        persistence = alpha + beta
        beta = torch.where(persistence >= 0.99, beta * 0.99 / persistence, beta)

        # Concatenate parameters
        output = torch.cat([alpha, beta, omega, gamma, lambda_], dim=1)
        return output
