from datetime import timedelta
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aurora import AuroraSmallPretrained, Batch, Metadata

### Configuration

DATASET_SURF_VARS = ["t2m", "u10", "v10", "msl"]  # surface variables to fine-tune
SURF_VARS = ["2t", "10u", "10v", "msl"]  # variables to include in the model
DATASET_EXTRA_SURF_VARS = ["ZWD"]  # additional surface variables to include
EXTRA_SURF_VARS = ["zwd"]  # lowercase for consistency in model input
ATMOS_VARS = ["z", "u", "v", "t", "q"]
STATIC_VARS = ["lsm", "z", "slt"]
LEV_SET = [100, 250, 500, 850]
H, W = 17, 32
T_IN = 2  # number of time context steps
LR = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data source (WeatherBench2 ERA5, adjust as needed)
# BUCKET_URL  = "gs://weatherbench2/datasets/era5/0.25deg/era5.zarr"


### Data Utils


def time_window(t, hours=6):
    return [t - timedelta(hours=hours * (T_IN - 1 - i)) for i in range(T_IN)]


class AuroraDataset(Dataset):
    def __init__(self, static_ds, atmos_ds, surd_ds, extra_surf_ds, times, device):
        self.static_ds = static_ds
        self.atmos_ds = atmos_ds
        self.surf_ds = surd_ds
        self.extra_ds = extra_surf_ds
        self.times = times
        self.device = device
        # Preload static vars (shared across samples)
        self.static = {
            v: torch.tensor(
                static_ds[v]
                .coarsen(
                    latitude=static_ds.sizes["latitude"] // H,
                    longitude=static_ds.sizes["longitude"] // W,
                    boundary="trim",
                )
                .mean()
                .values.astype("float32"),
                device=torch.device("cpu"),
            ).squeeze()
            for v in STATIC_VARS
        }

    def __len__(self):
        return len(self.times)

    def _coarsen(self, da):
        return torch.tensor(
            da.coarsen(
                latitude=da.sizes["latitude"] // H,
                longitude=da.sizes["longitude"] // W,
                boundary="trim",
            )
            .mean()
            .values.astype("float32"),
            device=torch.device("cpu"),
        )

    def __getitem__(self, i):
        t0 = self.times[i]
        ctx = time_window(t0)
        surf = {}
        for i, v in enumerate(DATASET_SURF_VARS):
            da = self.surf_ds[v].sel(valid_time=ctx).load()
            surf[SURF_VARS[i]] = self._coarsen(da).unsqueeze(0)
        for i, v in enumerate(DATASET_EXTRA_SURF_VARS):
            da = self.extra_ds[v].sel(time=ctx).load()
            surf[EXTRA_SURF_VARS[i]] = self._coarsen(da).unsqueeze(0)

        atmos = {}
        for v in ATMOS_VARS:
            da = (
                self.atmos_ds[v]
                .sel(valid_time=ctx, pressure_level=LEV_SET, method="nearest")
                .load()
            )
            ct = self._coarsen(da)
            ct = ct.permute(0, 2, 3, 1)  # (T, H, W, L)
            atmos[v] = ct.permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, L, H, W)

        md = Metadata(
            lat=torch.linspace(90, -90, H, device=torch.device("cpu")),
            lon=torch.linspace(0, 360, W + 1, device=torch.device("cpu"))[:-1],
            time=(t0,),
            atmos_levels=tuple(LEV_SET),
        )
        return Batch(surf_vars=surf, static_vars=self.static, atmos_vars=atmos, metadata=md)


def collate_batches(bs):
    out = bs[0]
    for b in bs[1:]:
        out = out.concatenate(b)
    return out


### Load Dataset & Split Times


# print("Opening dataset:", BUCKET_URL)
# ds = xr.open_zarr(BUCKET_URL, consolidated=True)
def main():
    # Ensure compatibility with DataLoader workers
    # Data will be downloaded here.
    print("Downloading data from:", "/home/space/data/ERA5/Aurora/")
    download_path = Path("/home/space/data/ERA5/Aurora/")

    zwd_path = Path("/home/space/data/ZWDX/era5/2023/")

    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_vars_ds = xr.open_dataset(download_path / "2023-01-surface-level.nc", engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(download_path / "2023-01-atmospheric.nc", engine="netcdf4")

    valid_suffixes = ("00.zwd.nc", "06.zwd.nc", "12.zwd.nc", "18.zwd.nc")

    files = sorted([f for f in zwd_path.rglob("*.zwd.nc") if f.name.endswith(valid_suffixes)])[
        :31
    ]  # Limit to 31 files for January

    zwd_vars_ds = xr.open_mfdataset(files, engine="netcdf4")
    print(f"Loaded {len(files)} ZWD files from {zwd_path}")

    # Combine datasets

    times = surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()
    available_times = set(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist())

    def is_valid_window(t):
        ctx = time_window(t)
        return all(ts in available_times for ts in ctx)

    times = [t for t in times if is_valid_window(t)]
    train_times = times[:20]  # First 24 time points for training
    val_times = times[20:24]  # Next 4 time points for validation
    # test_times = times[24:]  # Remaining time points for testing

    train_ds = AuroraDataset(
        static_vars_ds, atmos_vars_ds, surf_vars_ds, zwd_vars_ds, train_times, DEVICE
    )
    # Note: zwd_vars_ds is used for the extra surface variables (e.g., "zwd")
    val_ds = AuroraDataset(
        static_vars_ds, atmos_vars_ds, surf_vars_ds, zwd_vars_ds, val_times, DEVICE
    )
    # test_ds = AuroraDataset(
    #     static_vars_ds, atmos_vars_ds, surf_vars_ds, zwd_vars_ds, test_times, DEVICE
    # )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_batches
    )  # , persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_batches
    )

    # Model Setup

    torch.manual_seed(0)
    model = AuroraSmallPretrained(surf_vars=tuple(SURF_VARS + EXTRA_SURF_VARS)).to(DEVICE)
    model.load_checkpoint(strict=False)
    model.train()
    model.configure_activation_checkpointing()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.L1Loss()

    # Track weight change
    initial_key = next(k for k in model.state_dict() if "weight" in k)
    initial_wt = model.state_dict()[initial_key].clone()
    print("Tracking weight:", initial_key)

    # -----------------------------
    # Training & Validation Loop
    # -----------------------------
    for epoch in tqdm(range(NUM_EPOCHS), desc="Training Epochs"):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch)

            loss = torch.tensor(0.0, device=DEVICE)
            for v in SURF_VARS + EXTRA_SURF_VARS:
                p = pred.surf_vars[v]
                t = batch.surf_vars[v][:, :1, : p.shape[2], : p.shape[3]]
                loss = loss + loss_fn(p, t)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_train = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                pred = model(batch)
                for v in SURF_VARS:
                    p = pred.surf_vars[v]
                    t = batch.surf_vars[v][:, :1, : p.shape[2], : p.shape[3]]
                    val_loss += loss_fn(p, t).item()
        avg_val = val_loss / len(val_loader)

        tqdm.write(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} — train MAE: {avg_train:.4f} | val MAE: {avg_val:.4f}"
        )

    # Final Weight-Change Check

    final_wt = model.state_dict()[initial_key]
    change = (final_wt - initial_wt).abs().mean().item()
    print(f"\n✅ Average weight change in {initial_key}: {change:.6e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensure compatibility with DataLoader workers
    main()
