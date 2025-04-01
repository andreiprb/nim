import torch
import torch.nn as nn
import torch.nn.functional as F


class _SVCXORLayer(nn.Module):
    def __init__(self, svc_model):
        super(_SVCXORLayer, self).__init__()
        self.svc_model = svc_model

    def forward(self, x):
        x_np = x.detach().cpu().numpy()

        y_pred_bits = self.svc_model.predict(x_np)
        y_pred_tensor = torch.tensor(y_pred_bits, dtype=torch.float32, device=x.device)
        out = torch.cat([x, y_pred_tensor], dim=1)
        return out


class HybridAlphaZeroNet(nn.Module):
    def __init__(self, svc_model, policy_output_dim):
        super(HybridAlphaZeroNet, self).__init__()
        self.svc_layer = _SVCXORLayer(svc_model)

        self.shared = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(64, policy_output_dim)

        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.svc_layer(x)
        features = self.shared(x)
        policy_logits = self.policy_head(features)
        policy = F.softmax(policy_logits, dim=1)
        value = self.value_head(features)
        return policy, value
