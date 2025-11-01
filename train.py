import math
import torch
import torch.optim as optim
from neo_synthcore_v5.neo_synthcore_v5 import NeoSynthCoreV5, unified_loss_v5, theta, entropy_guided_kernel_selection
from config import CONFIG

def make_mock_batch(batch_size, seq_len, d_model, vocab_size, cultures):
x = torch.randint(0, vocab_size, (batch_size, seq_len))
env = torch.rand(batch_size, seq_len, d_model)
culture_id = torch.randint(0, cultures, (batch_size,))
targets = {
"lm": torch.randint(0, vocab_size, (batch_size, seq_len)),
"emo": torch.randint(0, CONFIG["emo_classes"], (batch_size,)),
"stage": torch.randint(0, CONFIG["stage_classes"], (batch_size,)),
"safe": torch.rand(batch_size, 1),
"culture": culture_id,
}
return x, env, culture_id, targets

def main():
cfg = CONFIG
device = torch.device(cfg["device"])
model = NeoSynthCoreV5(
d_model=cfg["d_model"],
vocab_size=cfg["vocab_size"],
emo_classes=cfg["emo_classes"],
stage_classes=cfg["stage_classes"],
cultures=cfg["cultures"],
).to(device)

opt = optim.AdamW(model.parameters(), lr=cfg["lr"])

# Example: use theta to adjust effective context length (k)
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
k = cfg["seq_len"]
usable_perf = theta(max(n_gpus, 1), k)
print(f"Usable performance (theta): {usable_perf:.4f}")

# Example: entropy-guided selection for kernel choice (mock costs)
costs = torch.tensor([0.8, 1.2, 0.9, 0.7])  # latency/FLOPs/memory proxies
H_opt, probs = entropy_guided_kernel_selection(costs)
print(f"Optimization entropy: {H_opt:.4f}, kernel probs: {probs}")

for epoch in range(cfg["epochs"]):
    x, env, culture_id, targets = make_mock_batch(cfg["batch_size"], cfg["seq_len"], cfg["d_model"], cfg["vocab_size"], cfg["cultures"])
    x, env, culture_id = x.to(device), env.to(device), culture_id.to(device)
    for k in targets:
        targets[k] = targets[k].to(device)

    outputs = model(x, env, culture_id)
    loss = unified_loss_v5(outputs, targets, cfg["lambdas"])

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

    # Risk-aware decoding example
    adj_logits = model.risk_aware_decode(outputs["lm_logits"], outputs["safety_score"])
    ppl = torch.exp(torch.mean(torch.logsumexp(adj_logits, dim=-1)))  # mock perplexity proxy

    print(f"Epoch {epoch+1}/{cfg['epochs']} | Loss: {loss.item():.4f} | PPL_proxy: {ppl.item():.2f}")

print("Training complete.")
if name == "main":
main()