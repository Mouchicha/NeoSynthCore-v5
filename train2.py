import torch
from neo.core import NeoSynthCoreV5
from neo.loss import unified_loss_v5
from config import CONFIG

def make_mock_batch(cfg):
x = torch.randint(0, cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]))
env = torch.rand(cfg["batch_size"], cfg["seq_len"], cfg["d_model"])
culture_id = torch.randint(0, cfg["cultures"], (cfg["batch_size"],))
targets = {
"lm": torch.randint(0, cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"])),
"emo": torch.randint(0, cfg["emo_classes"], (cfg["batch_size"],)),
"stage": torch.randint(0, cfg["stage_classes"], (cfg["batch_size"],)),
"safe": torch.rand(cfg["batch_size"], 1),
"culture": culture_id
}
return x, env, culture_id, targets

def train():
cfg = CONFIG
model = NeoSynthCoreV5(cfg).to(cfg["device"])
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

for epoch in range(cfg["epochs"]):
    x, env, culture_id, targets = make_mock_batch(cfg)
    x, env, culture_id = x.to(cfg["device"]), env.to(cfg["device"]), culture_id.to(cfg["device"])
    for k in targets: targets[k] = targets[k].to(cfg["device"])

    outputs = model(x, env, culture_id)
    loss = unified_loss_v5(outputs, targets, cfg["lambdas"])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
if name == "main":
train()