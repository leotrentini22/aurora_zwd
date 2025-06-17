from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from aurora import AuroraSmallPretrained, Batch, Metadata, rollout

# Data will be downloaded here.
download_path = Path("/home/space/data/ERA5/Aurora/")

zwd_path = Path("/home/space/data/ZWDX/era5/2023/")

static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2023-01-01-atmospheric.nc", engine="netcdf4")

valid_suffixes = ("00.zwd.nc", "06.zwd.nc", "12.zwd.nc", "18.zwd.nc")

files = sorted([f for f in zwd_path.rglob("*.zwd.nc") if f.name.endswith(valid_suffixes)])[:4]

zwd_vars_ds = xr.open_mfdataset(files, engine="netcdf4")

batch = Batch(
    surf_vars={
        # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
        # inserts a batch dimension of size one.
        "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),
        "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
        "10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
        "msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None]),
        "zwd": torch.from_numpy(zwd_vars_ds["ZWD"].values[:2][None]),
    },
    static_vars={
        # The static variables are constant, so we just get them for the first time.
        "z": torch.from_numpy(static_vars_ds["z"].values[0]),
        "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
        "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
    },
    atmos_vars={
        "t": torch.from_numpy(atmos_vars_ds["t"].values[:2][None]),
        "u": torch.from_numpy(atmos_vars_ds["u"].values[:2][None]),
        "v": torch.from_numpy(atmos_vars_ds["v"].values[:2][None]),
        "q": torch.from_numpy(atmos_vars_ds["q"].values[:2][None]),
        "z": torch.from_numpy(atmos_vars_ds["z"].values[:2][None]),
    },
    metadata=Metadata(
        lat=torch.from_numpy(surf_vars_ds.latitude.values),
        lon=torch.from_numpy(surf_vars_ds.longitude.values),
        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element. Select element 1, corresponding to time
        # 06:00.
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)

# model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
# model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")

model = AuroraSmallPretrained(
    surf_vars=("2t", "10u", "10v", "msl", "zwd"),
)
model.load_checkpoint(strict=False)

# hardcoded solution for the zwd variable
# with torch.no_grad():
#     model.encoder.surf_token_embeds.weights['zwd'] = torch.nn.Parameter(
#         torch.zeros_like(next(iter(model.encoder.surf_token_embeds.weights.values())))
#     )
# model.decoder.surf_heads['zwd'] = torch.nn.Linear(
#     in_features=next(iter(model.decoder.surf_heads.values())).in_features,
#     out_features=next(iter(model.decoder.surf_heads.values())).out_features
# )


model.eval()
# model = model.to("cuda")

with torch.inference_mode():
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]

# model = model.to("cpu")

fig, ax = plt.subplots(2, 3, figsize=(12, 9))  # 3 rows now

# for i in range(2):  # Adjust this to min(len(preds), 2) if unsure
#     pred = preds[i]
#     aurora_2t = pred.surf_vars["2t"][0, 0].numpy() - 273.15
#     era5_2t = surf_vars_ds["t2m"][2 + i].values - 273.15
#     era5_2t = era5_2t[1:,:]
#     error = np.abs(aurora_2t - era5_2t)

#     # Aurora Prediction
#     ax[i, 0].imshow(aurora_2t, vmin=-50, vmax=50, cmap='coolwarm')
#     ax[i, 0].set_title("Aurora Prediction" if i == 0 else "")
#     ax[i, 0].set_xticks([])
#     ax[i, 0].set_yticks([])

#     # ERA5 Ground Truth
#     ax[i, 1].imshow(era5_2t, vmin=-50, vmax=50, cmap='coolwarm')
#     ax[i, 1].set_title("ERA5" if i == 0 else "")
#     ax[i, 1].set_xticks([])
#     ax[i, 1].set_yticks([])

#     # Absolute Error
#     ax[i, 2].imshow(error, vmin=0, vmax=20, cmap='inferno')
#     ax[i, 2].set_title("Abs Error (°C)" if i == 0 else "")
#     ax[i, 2].set_xticks([])
#     ax[i, 2].set_yticks([])

for i in range(2):  # Adjust this to min(len(preds), 2) if unsure
    pred = preds[i]
    aurora_2t = pred.surf_vars["zwd"][0, 0].numpy() - 273.15
    era5_2t = zwd_vars_ds["ZWD"][2 + i].values - 273.15
    era5_2t = era5_2t[1:, :]
    error = np.abs(aurora_2t - era5_2t)

    # Aurora Prediction
    ax[i, 0].imshow(aurora_2t, vmin=-50, vmax=50, cmap="coolwarm")
    ax[i, 0].set_title("Aurora ZWD Prediction" if i == 0 else "")
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    # ERA5 Ground Truth
    ax[i, 1].imshow(era5_2t, vmin=-50, vmax=50, cmap="coolwarm")
    ax[i, 1].set_title("ERA5" if i == 0 else "")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

    # Absolute Error
    ax[i, 2].imshow(error, vmin=0, vmax=20, cmap="inferno")
    ax[i, 2].set_title("Abs Error (°C)" if i == 0 else "")
    ax[i, 2].set_xticks([])
    ax[i, 2].set_yticks([])

plt.tight_layout()
plt.show()
