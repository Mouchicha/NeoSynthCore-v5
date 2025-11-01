import torch
from neo_synthcore_v5.neo_synthcore_v5 import NeoSynthCoreV5, unified_loss_v5

def test_forward_and_loss():
model = NeoSynthCoreV5()
x = torch.randint(0, 10000, (8, 32))
env = torch.rand(8, 32, 256)
culture_id = torch.randint(0, 3, (8,))
outputs = model(x, env, culture_id)

targets = {
    "lm": torch.randint(0, 10000, (8, 32)),
    "emo": torch.randint(0, 6, (8,)),
    "stage": torch.randint(0, 5, (8,)),
    "safe": torch.rand(8, 1),
    "culture": culture_id,
}
lambdas = {"lm": 1.0, "emo": 0.5, "stage": 0.5, "safe": 1.0, "culture": 0.3, "vib": 0.1, "heal": 0.2}
loss = unified_loss_v5(outputs, targets, lambdas)
assert loss.item() > 0.0