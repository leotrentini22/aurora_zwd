import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from aurora import Aurora, Batch, Metadata, rollout

# Data will be downloaded here.
download_path = Path("/home/space/data/ERA5/Aurora/2023/")

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

zwd_batch = Batch(
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

model = Aurora(use_lora=False)  # The pretrained version does not use LoRA.
model.load_checkpoint(strict=False)

# model = AuroraSmallPretrained()
# model.load_checkpoint(strict=False)

model.eval()
# model = model.to("cuda")

with torch.inference_mode():
    print("Running inference...")
    start_time = time.time()
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_step = total_time / len(preds)

    print(f"Inference completed in {total_time:.2f} seconds.")
    print(f"Average time per prediction step: {avg_time_per_step:.2f} seconds.")

# model = model.to("cpu")

fig, ax = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)  # 2 rows, 3 columns

time_labels = ["2023-01-01 00:00 UTC", "2023-01-01 06:00 UTC"]

# For storing the images (to attach colorbars later)
im_pred, im_gt, im_err = None, None, None

for i in range(2):  # Loop over 2 timesteps
    pred = preds[i]
    aurora_2t = pred.surf_vars["2t"][0, 0].numpy() - 273.15
    era5_2t = surf_vars_ds["t2m"][2 + i].values - 273.15
    era5_2t = era5_2t[1:, :]

    # Squared relative error
    error = ((aurora_2t - era5_2t) / np.clip(np.abs(era5_2t), 1e-3, None)) ** 2

    # Aurora Prediction
    im_pred = ax[i, 0].imshow(aurora_2t, vmin=-50, vmax=50, cmap="coolwarm")
    ax[i, 0].set_ylabel(f"{time_labels[i]}\n2m Temp", fontsize=9)
    ax[i, 0].set_ylabel(time_labels[i], fontsize=10)
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    # ERA5 Ground Truth
    im_gt = ax[i, 1].imshow(era5_2t, vmin=-50, vmax=50, cmap="coolwarm")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

    # Squared Relative Error
    im_err = ax[i, 2].imshow(error, vmin=0, vmax=20, cmap="inferno")
    ax[i, 2].set_xticks([])
    ax[i, 2].set_yticks([])

# Set column titles
ax[0, 0].set_title("Aurora Prediction (°C)")
ax[0, 1].set_title("ERA5 Ground Truth (°C)")
ax[0, 2].set_title("Squared Relative Error")

# Add colorbars to the right of each column
cb1 = fig.colorbar(
    ScalarMappable(norm=Normalize(vmin=-50, vmax=50), cmap="coolwarm"),
    ax=ax[:, 0],
    fraction=0.02,
    pad=0.02,
    label="°C",
)

cb2 = fig.colorbar(
    ScalarMappable(norm=Normalize(vmin=-50, vmax=50), cmap="coolwarm"),
    ax=ax[:, 1],
    fraction=0.02,
    pad=0.02,
    label="°C",
)

cb3 = fig.colorbar(
    ScalarMappable(norm=Normalize(vmin=0, vmax=20), cmap="inferno"),
    ax=ax[:, 2],
    fraction=0.02,
    pad=0.02,
    label="(Δ/GT)²",
)

# Use only one suptitle
plt.suptitle("Aurora vs ERA5 Temperature Forecasts (2m Air Temp)", fontsize=14, y=1.02)

plt.savefig("../aurora_vs_era5_predictions.png", dpi=300, bbox_inches="tight")


################### Now let's also include ZWD in the predictions and compare it with ERA5 ZWD.

# model = AuroraSmallPretrained(
#     surf_vars=("2t", "10u", "10v", "msl", "zwd"),
# )
# model.load_checkpoint(strict=False)

model = Aurora(
    surf_vars=("2t", "10u", "10v", "msl", "zwd"),
    use_lora=False,  # The pretrained version does not use LoRA.
)
model.load_checkpoint(strict=False)

model.eval()
# model = model.to("cuda")

with torch.inference_mode():
    print("Running inference...")
    start_time = time.time()

    zwd_preds = [pred.to("cpu") for pred in rollout(model, zwd_batch, steps=2)]

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_step = total_time / len(zwd_preds)

    print(f"Inference completed in {total_time:.2f} seconds.")
    print(f"Average time per prediction step: {avg_time_per_step:.2f} seconds.")

# model = model.to("cpu")

fig, ax = plt.subplots(4, 3, figsize=(14, 10), constrained_layout=True)  # 4 rows, 3 columns

time_labels = ["2023-01-01 00:00 UTC", "2023-01-01 06:00 UTC"]

# For storing the images (to attach colorbars later)
im_pred, im_gt, im_err = None, None, None

for i in range(2):  # First two rows: 2m temperature
    pred = zwd_preds[i]
    aurora_2t = pred.surf_vars["2t"][0, 0].numpy() - 273.15
    era5_2t = surf_vars_ds["t2m"][2 + i].values - 273.15
    era5_2t = era5_2t[1:, :]
    error = ((aurora_2t - era5_2t) / np.clip(np.abs(era5_2t), 1e-3, None)) ** 2

    im_pred = ax[i, 0].imshow(aurora_2t, vmin=-50, vmax=50, cmap="coolwarm")
    ax[i, 0].set_ylabel(f"{time_labels[i]}\n2m Temp", fontsize=9)
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    im_gt = ax[i, 1].imshow(era5_2t, vmin=-50, vmax=50, cmap="coolwarm")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

    im_err = ax[i, 2].imshow(error, vmin=0, vmax=20, cmap="inferno")
    ax[i, 2].set_xticks([])
    ax[i, 2].set_yticks([])

for i in range(2, 4):  # Last two rows: ZWD
    pred = zwd_preds[i - 2]
    aurora_zwd = pred.surf_vars["zwd"][0, 0].numpy() - 273.15
    era5_zwd = zwd_vars_ds["ZWD"][i].values - 273.15
    era5_zwd = era5_zwd[1:, :]
    error = ((aurora_zwd - era5_zwd) / np.clip(np.abs(era5_zwd), 1e-3, None)) ** 2

    im_pred = ax[i, 0].imshow(aurora_zwd, vmin=-50, vmax=50, cmap="coolwarm")
    ax[i, 0].set_ylabel(f"{time_labels[i - 2]}\nZWD", fontsize=9)
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    im_gt = ax[i, 1].imshow(era5_zwd, vmin=-50, vmax=50, cmap="coolwarm")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

    im_err = ax[i, 2].imshow(error, vmin=0, vmax=20, cmap="inferno")
    ax[i, 2].set_xticks([])
    ax[i, 2].set_yticks([])

# Set column titles
ax[0, 0].set_title("Aurora Prediction (°C)")
ax[0, 1].set_title("ERA5 Ground Truth (°C)")
ax[0, 2].set_title("Squared Relative Error")

# Colorbars for temperature (first 2 rows)
cb1 = fig.colorbar(
    ScalarMappable(norm=Normalize(vmin=-50, vmax=50), cmap="coolwarm"),
    ax=ax[0:2, 0],
    fraction=0.02,
    pad=0.02,
    label="2m Temp (°C)",
)

cb2 = fig.colorbar(
    ScalarMappable(norm=Normalize(vmin=-50, vmax=50), cmap="coolwarm"),
    ax=ax[0:2, 1],
    fraction=0.02,
    pad=0.02,
    label="2m Temp (°C)",
)

# Colorbars for ZWD (last 2 rows)
cb3 = fig.colorbar(
    ScalarMappable(norm=Normalize(vmin=-50, vmax=50), cmap="coolwarm"),
    ax=ax[2:4, 0],
    fraction=0.02,
    pad=0.02,
    label="ZWD (°C-equivalent)",
)

cb4 = fig.colorbar(
    ScalarMappable(norm=Normalize(vmin=-50, vmax=50), cmap="coolwarm"),
    ax=ax[2:4, 1],
    fraction=0.02,
    pad=0.02,
    label="ZWD (°C-equivalent)",
)

# Colorbar for all error plots (all rows)
cb5 = fig.colorbar(
    ScalarMappable(norm=Normalize(vmin=0, vmax=20), cmap="inferno"),
    ax=ax[:, 2],
    fraction=0.02,
    pad=0.02,
    label="Squared Relative Error",
)

# Figure title
plt.suptitle(
    "Aurora vs ERA5: Temperature (2m) and ZWD Forecasts on 2023-01-01", fontsize=14, y=1.02
)

plt.savefig("../aurora_vs_era5_predictions_with_zwd.png", dpi=300, bbox_inches="tight")
