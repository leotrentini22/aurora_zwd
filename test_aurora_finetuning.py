from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from aurora import AuroraSmallPretrained, Batch, Metadata

# --- Reproducibility ---
torch.manual_seed(0)

# --- Settings ---
target_vars = ("2t", "10u", "10v", "msl", "zwd")
learning_rate = 1e-4
num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Dummy Batch Generation ---
def get_dummy_batch():
    H, W = 17, 32  # spatial resolution
    return Batch(
        surf_vars={k: torch.randn(1, 2, H, W).to(device) for k in target_vars},
        static_vars={k: torch.randn(H, W).to(device) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, H, W).to(device) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, H).to(device),
            lon=torch.linspace(0, 360, W + 1)[:-1].to(device),
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )


# --- Model Setup ---
model = AuroraSmallPretrained(surf_vars=target_vars)
model.load_checkpoint(strict=False)  # Load pretrained weights without strict matching
model = model.to(device)
model.train()
model.configure_activation_checkpointing()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.L1Loss()  # MAE

# --- Save initial weights for verification ---
initial_key = next(k for k in model.state_dict() if "weight" in k)
initial_weight = model.state_dict()[initial_key].clone()
print(f"Tracking weight change for: {initial_key}")


# --- Training Loop ---
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    batch = get_dummy_batch()
    optimizer.zero_grad()

    pred = model.forward(batch)
    total_loss = None

    for var in target_vars:
        prediction = pred.surf_vars[var]
        target = batch.surf_vars[var]
        target = target[:, :, : prediction.shape[2], : prediction.shape[3]]  # crop height/width

        loss = loss_fn(prediction, target)
        total_loss = loss if total_loss is None else total_loss + loss

        print(f"  [{var}] MAE: {loss.item():.4f}  |  pred mean: {prediction.mean().item():.4f}")

    assert total_loss is not None
    total_loss.backward()

    # Gradient check
    grad_norm = None
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  First grad norm (param: {name}): {grad_norm:.4e}")
            break

    optimizer.step()
    print(f"  Total Loss: {total_loss.item():.4f}")

# --- Final weight change verification ---
final_weight = model.state_dict()[initial_key]
weight_change = (final_weight - initial_weight).abs().mean().item()
print(f"\n✅ Average weight change in {initial_key}: {weight_change:.6e}")
