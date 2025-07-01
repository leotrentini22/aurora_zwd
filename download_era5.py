################## EXAMPLE USAGE ##################
# # Default: download both types for 2010–2024
# python download_era5.py

# # Specific year(s) and both data types
# python download_era5.py 2021 2023

# # Range and only surface-level data
# python download_era5.py 2019-2020 --surface-only

# # Only pressure-level data for all years
# python download_era5.py --pressure-only

###################################################

import argparse
import re
from pathlib import Path

import cdsapi

c = cdsapi.Client()


# ---------- CLI ---------
def parse_year_tokens(tokens):
    """
    Expand tokens like 2020, 2018-2020, 1999‑2001 into a sorted list of ints.
    """
    years = []
    for tok in tokens:
        if re.fullmatch(r"\d{4}-\d{4}", tok):  # range 2010-2015
            start, end = map(int, tok.split("-"))
            if end < start:
                raise ValueError(f"Bad range {tok}: end < start")
            years.extend(range(start, end + 1))
        elif re.fullmatch(r"\d{4}", tok):  # single year 2023
            years.append(int(tok))
        else:
            raise ValueError(f"Unrecognised year token: {tok}")
    return sorted(set(years))  # de‑duplicate & sort


parser = argparse.ArgumentParser(description="Download ERA5 surface and pressure-level data.")
parser.add_argument(
    "years",
    nargs="*",
    metavar="YEAR",
    help="Optional list of years or ranges (e.g. 2020 2022-2023). Defaults to 2010-2024.",
)

group = parser.add_mutually_exclusive_group()
group.add_argument("--surface-only", action="store_true", help="Only download surface-level data")
group.add_argument("--pressure-only", action="store_true", help="Only download pressure-level data")

args = parser.parse_args()
years = parse_year_tokens(args.years) if args.years else list(range(2010, 2025))
download_surface = not args.pressure_only
download_pressure = not args.surface_only
# ---------------------------------


# Data will be downloaded here.
download_path = Path("/home/space/data/ERA5/Aurora/")
download_path = download_path.expanduser()
download_path.mkdir(parents=True, exist_ok=True)


# Download the static variables.
if not (download_path / "static.nc").exists():
    print("Downloading static variables...")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "geopotential",
                "land_sea_mask",
                "soil_type",
            ],
            "year": "2023",
            "month": "01",
            "day": "01",
            "time": "00:00",
            "format": "netcdf",
        },
        str(download_path / "static.nc"),
    )
    print("Static variables downloaded!")


# Download the surface-level variables.
# if not (download_path / "2023-01-01-surface-level.nc").exists():
#     c.retrieve(
#         "reanalysis-era5-single-levels",
#         {
#             "product_type": "reanalysis",
#             "variable": [
#                 "2m_temperature",
#                 "10m_u_component_of_wind",
#                 "10m_v_component_of_wind",
#                 "mean_sea_level_pressure",
#             ],
#             "year": "2023",
#             "month": "01",
#             "day": "01",
#             "time": ["00:00", "06:00", "12:00", "18:00"],
#             "format": "netcdf",
#         },
#         str(download_path / "2023-01-01-surface-level.nc"),
#     )

for year in years:
    for month in range(1, 13):
        filename = f"{year:04d}-{month:02d}-surface-level.nc"
        if not (download_path / filename).exists() and download_surface:
            print(f"Downloading {filename}...")
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "2m_temperature",
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "mean_sea_level_pressure",
                    ],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "format": "netcdf",
                },
                str(download_path / filename),
            )
            print(f"Downloaded {filename}")

        # print("Surface-level variables downloaded!")

        # Download the atmospheric variables.
        filename = f"{year:04d}-{month:02d}-atmospheric.nc"
        if not (download_path / filename).exists() and download_pressure:
            print(f"Downloading {filename}...")
            c.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "temperature",
                        "u_component_of_wind",
                        "v_component_of_wind",
                        "specific_humidity",
                        "geopotential",
                    ],
                    "pressure_level": [
                        "50",
                        "100",
                        "150",
                        "200",
                        "250",
                        "300",
                        "400",
                        "500",
                        "600",
                        "700",
                        "850",
                        "925",
                        "1000",
                    ],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "format": "netcdf",
                },
                str(download_path / filename),
            )
            print(f"Downloaded {filename}")

# if not (download_path / "2023-01-01-atmospheric.nc").exists():
#     c.retrieve(
#         "reanalysis-era5-pressure-levels",
#         {
#             "product_type": "reanalysis",
#             "variable": [
#                 "temperature",
#                 "u_component_of_wind",
#                 "v_component_of_wind",
#                 "specific_humidity",
#                 "geopotential",
#             ],
#             "pressure_level": [
#                 "50",
#                 "100",
#                 "150",
#                 "200",
#                 "250",
#                 "300",
#                 "400",
#                 "500",
#                 "600",
#                 "700",
#                 "850",
#                 "925",
#                 "1000",
#             ],
#             "year": "2023",
#             "month": "01",
#             "day": "01",
#             "time": ["00:00", "06:00", "12:00", "18:00"],
#             "format": "netcdf",
#         },
#         str(download_path / "2023-01-01-atmospheric.nc"),
#     )
# print("Atmospheric variables downloaded!")
