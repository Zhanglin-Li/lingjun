"""PyTorch model definitions for simplified training script."""
import torch
import torch.nn as nn


def r2_weighted_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: torch.Tensor = None
) -> torch.Tensor:
    """Compute the weighted R² score using PyTorch tensors.

    Args:
        y_true (torch.Tensor): Ground truth tensor.
        y_pred (torch.Tensor): Predicted tensor.
        sample_weight (torch.Tensor, optional): Weights for each observation. Defaults to None.

    Returns:
        torch.Tensor: Weighted R² score.
    """
    if sample_weight is None:
        sample_weight = torch.ones_like(y_true)
    numerator = torch.sum(sample_weight * (y_pred - y_true) ** 2)
    denominator = torch.sum(sample_weight * (y_true) ** 2) + 1e-38
    r2 = 1 - (numerator / denominator)
    return r2


class WeightedR2Loss(nn.Module):
    """PyTorch loss function for weighted R²."""
    def __init__(self, epsilon: float = 1e-38) -> None:
        super(WeightedR2Loss, self).__init__()
        self.epsilon = epsilon

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute the weighted R² loss.

        Args:
            y_true (torch.Tensor): Ground truth tensor.
            y_pred (torch.Tensor): Predicted tensor.
            weights (torch.Tensor, optional): Weights for each observation. Defaults to None.

        Returns:
            torch.Tensor: Computed weighted R² loss.
        """
        if weights is None:
            weights = torch.ones_like(y_true)
        numerator = torch.sum(weights * (y_pred - y_true) ** 2)
        denominator = torch.sum(weights * (y_true) ** 2) + self.epsilon
        loss = numerator / denominator
        return loss


class ModelRBase(nn.Module):
    """Base recurrent model with GRU/LSTM layers + fully connected layers."""
    def __init__(self, input_size, hidden_sizes, dropout_rates, hidden_sizes_linear, dropout_rates_linear, model_type):
        super().__init__()
        self.num_layers = len(hidden_sizes)
        self.gru_layers = nn.ModuleList()
        self.dropout_rates = nn.ModuleList()

        for i in range(self.num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i - 1]
            layer = nn.GRU(input_dim, hidden_sizes[i], num_layers=1, batch_first=True) if model_type == "gru" else \
                    nn.LSTM(input_dim, hidden_sizes[i], num_layers=1, batch_first=True)
            self.gru_layers.append(layer)
            self.dropout_rates.append(nn.Dropout(dropout_rates[i]))

        n_input_linear = input_size if self.num_layers == 0 else hidden_sizes[-1]
        fc_layers = []
        if hidden_sizes_linear:
            for i, h in enumerate(hidden_sizes_linear):
                in_feat = n_input_linear if i == 0 else hidden_sizes_linear[i - 1]
                fc_layers.extend([nn.Linear(in_feat, h), nn.ReLU(), nn.Dropout(dropout_rates_linear[i])])
            fc_layers.append(nn.Linear(hidden_sizes_linear[-1], 1))
        else:
            fc_layers.append(nn.Linear(n_input_linear, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x, hidden=None, return_raw=False):
        D, T, _ = x.shape
        if hidden is None:
            hidden = [None] * self.num_layers
        for i, gru in enumerate(self.gru_layers):
            x, h = gru(x, hidden[i])
            x = self.dropout_rates[i](x)
            hidden[i] = h
        if return_raw:
            return x, hidden
        x = x.reshape(D * T, -1)
        x = self.fc(x)
        x = x.reshape(D, T)
        return x, hidden


class ModelR(nn.Module):
    """Recurrent model with auxiliary heads for multi-task learning.

    Architecture:
    - Multiple GRU/LSTM layers for each responder
    - Auxiliary heads for each responder (用于辅助任务)
    - Main output head that concatenates GRU outputs
    """
    def __init__(self, input_size, hidden_sizes, dropout_rates, hidden_sizes_linear, dropout_rates_linear, model_type):
        super().__init__()
        self.num_resp = 2  # 处理 2 个 responders
        self.hidden_size = hidden_sizes[-1] if hidden_sizes else input_size

        # GRU layers for each responder
        self.grus = nn.ModuleList()
        for _ in range(self.num_resp):
            self.grus.append(ModelRBase(input_size, hidden_sizes, dropout_rates, hidden_sizes_linear, dropout_rates_linear, model_type))

        # Main output head
        self.main_out = nn.Linear(self.hidden_size * self.num_resp, 1)

        # Auxiliary heads for each responder
        self.aux_heads = nn.ModuleList()
        for _ in range(self.num_resp):
            aux_head = nn.Sequential()
            if hidden_sizes_linear:
                for i, h in enumerate(hidden_sizes_linear):
                    in_feat = self.hidden_size if i == 0 else hidden_sizes_linear[i-1]
                    aux_head.add_module(f'aux_fc_{i}', nn.Linear(in_feat, h))
                    aux_head.add_module(f'aux_relu_{i}', nn.ReLU())
                    aux_head.add_module(f'aux_drop_{i}', nn.Dropout(dropout_rates_linear[i]))
                aux_head.add_module('aux_out', nn.Linear(hidden_sizes_linear[-1], 1))
            else:
                aux_head.add_module('aux_out', nn.Linear(self.hidden_size, 1))
            self.aux_heads.append(aux_head)

    def forward(self, x, hidden=None, use_aux_heads=True):
        D, T, _ = x.shape
        if hidden is None:
            hidden = [None] * self.num_resp

        gru_outputs = []
        new_hidden = []
        for i in range(self.num_resp):
            gru_out, h = self.grus[i](x, hidden[i], return_raw=True)
            gru_outputs.append(gru_out)
            new_hidden.append(h)

        concat_out = torch.cat(gru_outputs, dim=-1)
        concat_flat = concat_out.reshape(D * T, -1)
        main_pred = self.main_out(concat_flat).reshape(D, T)

        aux_preds = None
        if use_aux_heads:
            aux_list = []
            for i in range(self.num_resp):
                gru_flat = gru_outputs[i].reshape(D * T, -1)
                aux_pred = self.aux_heads[i](gru_flat).reshape(D, T, 1)
                aux_list.append(aux_pred)
            aux_preds = torch.cat(aux_list, dim=-1)

        return main_pred, aux_preds, new_hidden
