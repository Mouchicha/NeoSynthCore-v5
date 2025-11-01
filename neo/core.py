import torch
import torch.nn as nn
import torch.nn.functional as F

class NeoSynthCoreV5(nn.Module):
def init(self, config):
super().init()
self.config = config
D = config['d_model']
V = config['vocab_size']
C = config['cultures']
E = config['emo_classes']
S = config['stage_classes']

    self.embedding = nn.Embedding(V, D)
    self.env_proj = nn.Linear(D, D)
    self.culture_embed = nn.Embedding(C, D)

    self.cognitive_tower = nn.TransformerEncoderLayer(D, nhead=4, batch_first=True)
    self.emotional_tower = nn.TransformerEncoderLayer(D, nhead=4, batch_first=True)

    self.gate_fc = nn.Linear(D * 3, 1)

    self.lm_head = nn.Linear(D, V)
    self.emo_head = nn.Linear(D, E)
    self.stage_head = nn.Linear(D, S)
    self.safety_head = nn.Linear(D, 1)
    self.culture_head = nn.Linear(D, C)

    self.posterior = nn.Linear(D, D)
    self.prior = nn.Linear(D, D)

def forward(self, x, env, culture_id):
    x_embed = self.embedding(x)
    env_embed = self.env_proj(env)
    culture_vec = self.culture_embed(culture_id).unsqueeze(1).expand_as(x_embed)

    c_out = self.cognitive_tower(x_embed + culture_vec)
    e_out = self.emotional_tower(x_embed + env_embed + culture_vec)

    q_t = torch.mean(env_embed, dim=1)
    gate_input = torch.cat([c_out.mean(1), e_out.mean(1), q_t], dim=1)
    G_t = torch.sigmoid(self.gate_fc(gate_input)).unsqueeze(1)

    fused = G_t * e_out + (1 - G_t) * c_out
    pooled = fused.mean(1)

    return {
        "lm_logits": self.lm_head(fused),
        "emo_logits": self.emo_head(pooled),
        "stage_logits": self.stage_head(pooled),
        "safety_score": torch.sigmoid(self.safety_head(pooled)),
        "culture_logits": self.culture_head(pooled),
        "kl_div": F.kl_div(
            F.log_softmax(self.posterior(pooled), dim=-1),
            F.softmax(self.prior(env_embed.mean(1)), dim=-1),
            reduction="batchmean"
        ),
        "fused": fused
    }