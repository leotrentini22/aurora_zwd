import random
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm  # progress bar

zwd_path = Path("/home/space/data/ZWDX/era5/")
valid_suffixes = ("00.zwd.nc", "06.zwd.nc", "12.zwd.nc", "18.zwd.nc")

files = sorted([f for f in zwd_path.rglob("*.zwd.nc") if f.name.endswith(valid_suffixes)])
print(f"Found {len(files)} files matching the criteria.")

random.shuffle(files)  # Shuffle the files to ensure randomness

n_total = 0
mean = 0.0
M2 = 0.0  # Sum of squared differences from the current mean (for variance)
count = 0

# Use tqdm for progress bar
with tqdm(files, desc="Processing files", unit="file") as pbar:
    for file in pbar:
        with xr.open_dataset(file) as ds:
            zwd = ds.ZWD.values.astype(np.float64)
            zwd = zwd[np.isfinite(zwd)]  # avoid NaNs

            n = zwd.size
            new_mean = zwd.mean()
            delta = new_mean - mean

            total = n_total + n
            mean += delta * n / total
            M2 += zwd.var(ddof=0) * n + (delta**2) * n_total * n / total
            n_total = total
            pbar.set_postfix(mean=f"{mean:.4f}", std=f"{np.sqrt(M2 / n_total):.4f}")
            if delta**2 / mean < 1e-4:  # Check if the change is significant
                count += 1
                if count > 5:  # If we have seen 10 small changes, stop
                    print(f"Early stopping after {count} iterations due to small changes.")
                    break
            else:
                count = 0

std = np.sqrt(M2 / n_total)

print(f"\nEmpirical Mean ZWD: {mean:.4f} mm")
print(f"Empirical Std ZWD: {std:.4f} mm")
