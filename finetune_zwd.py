from datetime import datetime

import torch

from aurora import AuroraSmallPretrained, Batch, Metadata

model = AuroraSmallPretrained()
model.load_checkpoint()

batch = Batch(
    surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 721),
        lon=torch.linspace(0, 360, 1440 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)

model = model.cuda()
model.train()
model.configure_activation_checkpointing()

# pred = model.forward(batch)
# loss = ...
# loss.backward()
