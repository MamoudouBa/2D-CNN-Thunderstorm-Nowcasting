#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
"""
Author: Mamoudou Ba
Refactored by GEMINI: Two-Pass Zarr Generator for Training and Validation (21 Features)
                      - Fits and saves joblib StandardScaler on Training Data (2016-2020, 2023-2024)
                      - Populates Training Zarr Stores
                      - Applies fitted StandardScaler to Validation Data (2025) and Populates Validation Zarr Stores
"""

import glob
import math
import os
from datetime import datetime
from pathlib import Path

import joblib
import netCDF4
import numpy as np
import pygrib
import zarr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

K_MINUS_C = 273.15
nx, ny = (501, 251)
nxny = nx * ny

BASE_DIR = Path("/contrib/Mamoudou.Ba").resolve()
SCALER_SAVE_PATH = BASE_DIR / "cnn_scaler_21var.joblib"

# Clean up and recreate Zarr stores
train_in_zarr = 'zarr_input_training_data_21_var_10km_f01_f08'
train_lbl_zarr = 'zarr_label_training_data_21_var_10km_f01_f08'
val_in_zarr = 'zarr_input_validation_data_21_var_10km_f01_f08'
val_lbl_zarr = 'zarr_label_validation_data_21_var_10km_f01_f08'

for z_path in [train_in_zarr, train_lbl_zarr, val_in_zarr, val_lbl_zarr]:
    if os.path.exists(z_path):
        os.system(f"rm -rf {z_path}")

training_set_21_var = zarr.open_array(
    train_in_zarr, mode='w', shape=(0, ny, nx, 21), chunks=(5, ny, nx, 21), dtype=np.float32
)
label_train_21_var = zarr.open_array(
    train_lbl_zarr, mode='w', shape=(0, ny, nx, 2), chunks=(5, ny, nx, 2), dtype=np.float32
)
validation_set_21_var = zarr.open_array(
    val_in_zarr, mode='w', shape=(0, ny, nx, 21), chunks=(5, ny, nx, 21), dtype=np.float32
)
label_val_21_var = zarr.open_array(
    val_lbl_zarr, mode='w', shape=(0, ny, nx, 2), chunks=(5, ny, nx, 2), dtype=np.float32
)

number_of_day_in_a_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

lead_hour_configs = [
    (1, "01"), (2, "02"), (3, "03"), (4, "04"),
    (5, "05"), (6, "06"), (7, "07"), (8, "08"),
]


def extract_raw_21_predictors(file_path) -> np.ndarray:
    """Extracts raw 21 HRRR predictors into shape (NY, NX, 21)."""
    hrrr_fields = pygrib.open(str(file_path))

    try:
        vvel_500mb = hrrr_fields.select(name='Vertical velocity', level=500)[0].values
        u_600mb    = hrrr_fields.select(name='U component of wind', level=600)[0].values
        v_600mb    = hrrr_fields.select(name='V component of wind', level=600)[0].values

        vvel_700mb = hrrr_fields.select(name='Vertical velocity', level=700)[0].values
        u_800mb    = hrrr_fields.select(name='U component of wind', level=800)[0].values
        v_800mb    = hrrr_fields.select(name='V component of wind', level=800)[0].values

        rh_850mb   = hrrr_fields.select(name='Relative humidity', level=850)[0].values
        spfh_850mb = hrrr_fields.select(name='Specific humidity', level=850)[0].values
        u_850mb    = hrrr_fields.select(name='U component of wind', level=850)[0].values
        v_850mb    = hrrr_fields.select(name='V component of wind', level=850)[0].values
        vvel_925mb = hrrr_fields.select(name='Vertical velocity', level=925)[0].values
        rh_1013mb  = hrrr_fields.select(name='Relative humidity', level=1013)[0].values
        soilw      = hrrr_fields.select(name='Volumetric soil moisture content')[0].values
        refc       = hrrr_fields.select(name='Maximum/Composite reflectivity')[0].values

        mslma      = hrrr_fields.select(name='Mean sea level pressure')[0].values
        surf_pres  = hrrr_fields.select(name='Surface pressure')[0].values / 100.0
        temp_2m    = hrrr_fields.select(name='2 metre temperature')[0].values - K_MINUS_C
        spfh_2m    = hrrr_fields.select(name='2 metre specific humidity')[0].values
        dewpoint   = hrrr_fields.select(name='2 metre dewpoint temperature')[0].values - K_MINUS_C

        u_surf     = hrrr_fields.select(name='10 metre U wind component')[0].values
        v_surf     = hrrr_fields.select(name='10 metre V wind component')[0].values

        li         = hrrr_fields.select(name='Lifted index')[0].values
        cape       = hrrr_fields.select(name='Convective available potential energy')[0].values
        cin        = hrrr_fields.select(name='Convective inhibition')[0].values
        pwat       = hrrr_fields.select(name='Precipitable water')[0].values

    except Exception:
        vvel_500mb = hrrr_fields.message(1).values[:, :]
        u_600mb    = hrrr_fields.message(2).values[:, :]
        v_600mb    = hrrr_fields.message(3).values[:, :]
        vvel_700mb = hrrr_fields.message(4).values[:, :]
        u_800mb    = hrrr_fields.message(5).values[:, :]
        v_800mb    = hrrr_fields.message(6).values[:, :]
        rh_850mb   = hrrr_fields.message(7).values[:, :]
        spfh_850mb = hrrr_fields.message(8).values[:, :]
        u_850mb    = hrrr_fields.message(9).values[:, :]
        v_850mb    = hrrr_fields.message(10).values[:, :]
        vvel_925mb = hrrr_fields.message(11).values[:, :]
        rh_1013mb  = hrrr_fields.message(12).values[:, :]
        soilw      = hrrr_fields.message(13).values[:, :]
        refc       = hrrr_fields.message(14).values[:, :]
        mslma      = hrrr_fields.message(15).values[:, :]
        surf_pres  = hrrr_fields.message(16).values[:, :] / 100.0
        temp_2m    = hrrr_fields.message(17).values[:, :] - K_MINUS_C
        spfh_2m    = hrrr_fields.message(18).values[:, :]
        dewpoint   = hrrr_fields.message(19).values[:, :] - K_MINUS_C
        u_surf     = hrrr_fields.message(20).values[:, :]
        v_surf     = hrrr_fields.message(21).values[:, :]
        li         = hrrr_fields.message(22).values[:, :]
        cape       = hrrr_fields.message(23).values[:, :]
        cin        = hrrr_fields.message(24).values[:, :]
        pwat       = hrrr_fields.message(25).values[:, :]

    hrrr_fields.close()

    v_600_spd = np.sqrt(u_600mb**2 + v_600mb**2)
    v_800_spd = np.sqrt(u_800mb**2 + v_800mb**2)
    v_surf_spd = np.sqrt(u_surf**2 + v_surf**2)

    alt_surf = ((1.0 - (surf_pres / 1013.25)**0.190284) * 145366.45) / 3.2808
    alt_800  = ((1.0 - (800.0 / 1013.25)**0.190284) * 145366.45) / 3.2808
    alt_600  = ((1.0 - (600.0 / 1013.25)**0.190284) * 145366.45) / 3.2808

    dist_surf_800 = np.maximum(alt_800 - alt_surf, 10.0)
    dist_600_800  = np.maximum(alt_600 - alt_800, 10.0)

    vshear_surf_800 = np.nan_to_num(((v_800_spd - v_surf_spd) / dist_surf_800) * 1000.0)
    vshear_600_800  = np.nan_to_num(((v_600_spd - v_800_spd) / dist_600_800) * 1000.0)

    raw_list = [
        pwat, mslma, rh_850mb, cape, cin, li,
        u_surf, v_surf, u_850mb, v_850mb, vvel_925mb, vvel_700mb,
        spfh_2m, u_800mb, v_800mb, u_600mb, v_600mb, vvel_500mb,
        vshear_surf_800, vshear_600_800,refc
    ]

    raw_list_transposed = [arr.T if arr.shape == (nx, ny) else arr for arr in raw_list]
    predictors_3d = np.stack(raw_list_transposed, axis=-1).astype(np.float32)
    return np.nan_to_num(predictors_3d, nan=0.0, posinf=0.0, neginf=0.0)


def get_hrrr_path(date_str, hour_str, lead_str, offset_hrs):
    """Calculates analysis datetime and resolves HRRR file path."""
    year_str, month_str, day_str = date_str[0:4], date_str[4:6], date_str[6:8]
    imonth, iyear = int(month_str), int(year_str)
    day_analysis_time_int = int(day_str)
    analysis_time_int = int(hour_str) - offset_hrs

    if analysis_time_int < 0:
        if day_analysis_time_int == 1:
            day_analysis_time_int = number_of_day_in_a_month[imonth - 1]
            imonth -= 1
            if imonth == 0:
                imonth = 12
                iyear -= 1
                day_analysis_time_int = number_of_day_in_a_month[imonth - 1]
            date_analysis_time_str = f"{iyear:04d}{imonth:02d}{day_analysis_time_int:02d}"
        else:
            date_analysis_time_str = f"{year_str}{month_str}{(int(day_str)-1):02d}"
        analysis_time_str = f"{(24 + analysis_time_int):02d}"
    else:
        analysis_time_str = f"{analysis_time_int:02d}"
        date_analysis_time_str = date_str

    path2 = BASE_DIR / f"hrrr_501_251/{date_analysis_time_str}/{date_analysis_time_str}_t{analysis_time_str}zprsf{lead_str}.grib2"
    return path2 if path2.exists() else None


# =========================================================================
# PASS 1: FIT STANDARD SCALER ON TRAINING DATA ONLY
# =========================================================================
print("\n" + "=" * 70)
print("PASS 1: Fitting StandardScaler on Training Data (2016-2020, 2023-2024)...")
print("=" * 70)

global_scaler = StandardScaler()
fit_count = 0

for offset_hrs, lead_str in lead_hour_configs:
    for file_path in sorted(glob.glob('mrms_501_251/20*/*.grib2')):
        url_str = file_path.split('/')
        date_str = url_str[1]
        time_str = url_str[2].split('_')[3].split('-')
        hour_str, mrms_min = time_str[1][0:2], time_str[1][2:4]

        if int(mrms_min) > 15:
            continue

        year_str, month_str = date_str[0:4], date_str[4:6]
        year_int, month_int = int(year_str), int(month_str)

        # 🌟 TRAINING YEARS ONLY for Pass 1
        if year_int in [2021, 2022, 2025] or month_int < 3 or month_int > 10:
            continue

        hrrr_file = get_hrrr_path(date_str, hour_str, lead_str, offset_hrs)
        if not hrrr_file or hrrr_file.stat().st_size < 2000000:
            continue

        try:
            mrms = pygrib.open(file_path)
            mrms_dbz = mrms.message(1).values[:, :]
            mrms.close()

            if (np.count_nonzero((mrms_dbz >= 35.0) & (mrms_dbz < 95.0)) / nxny) * 100.0 < 0.5:
                continue

            raw_3d = extract_raw_21_predictors(hrrr_file)
            global_scaler.partial_fit(raw_3d.reshape(-1, 21))
            fit_count += 1
            if fit_count % 50 == 0:
                print(f"Partial fit iteration {fit_count} completed...")

        except Exception as e:
            continue

joblib.dump(global_scaler, SCALER_SAVE_PATH)
print(f"🎉 StandardScaler fitted on {fit_count} training cases and saved to {SCALER_SAVE_PATH}")


# =========================================================================
# PASS 2 & 3: TRANSFORM AND POPULATE TRAINING & VALIDATION ZARR STORES
# =========================================================================
print("\n" + "=" * 70)
print("PASS 2 & 3: Populating Training and Validation Zarr Stores...")
print("=" * 70)

train_samples, val_samples = 0, 0

for offset_hrs, lead_str in lead_hour_configs:
    for file_path in sorted(glob.glob('mrms_501_251/20*/*.grib2')):
        url_str = file_path.split('/')
        date_str = url_str[1]
        time_str = url_str[2].split('_')[3].split('-')
        hour_str, mrms_min = time_str[1][0:2], time_str[1][2:4]

        if int(mrms_min) > 15:
            continue

        year_str, month_str = date_str[0:4], date_str[4:6]
        year_int, month_int = int(year_str), int(month_str)

        if month_int < 3 or month_int > 10:
            continue

        # Skip excluded baseline validation years 2021-2022
        if year_int in [2021, 2022]:
            continue

        hrrr_file = get_hrrr_path(date_str, hour_str, lead_str, offset_hrs)
        if not hrrr_file or hrrr_file.stat().st_size < 2000000:
            continue

        try:
            mrms = pygrib.open(file_path)
            mrms_dbz = mrms.message(1).values[:, :]
            mrms.close()

            if (np.count_nonzero((mrms_dbz >= 35.0) & (mrms_dbz < 95.0)) / nxny) * 100.0 < 0.5:
                continue

            raw_3d = extract_raw_21_predictors(hrrr_file)
            scaled_flat = global_scaler.transform(raw_3d.reshape(-1, 21))
            scaled_3d = scaled_flat.reshape(ny, nx, 21).astype(np.float32)

            tstorm = np.where((mrms_dbz >= 35.0) & (mrms_dbz < 95.0), 1.0, 0.0)
            label_sample = to_categorical(tstorm, num_classes=2).astype(np.float32)

            if year_int == 2025:
                # 🌟 POPULATE VALIDATION ZARR (2025)
                validation_set_21_var.append(np.expand_dims(scaled_3d, axis=0))
                label_val_21_var.append(np.expand_dims(label_sample, axis=0))
                val_samples += 1
                print(f"Validation Sample {val_samples}: {file_path} | HRRR: {hrrr_file.name}")
            else:
                # 🌟 POPULATE TRAINING ZARR (2016-2020, 2023-2024)
                training_set_21_var.append(np.expand_dims(scaled_3d, axis=0))
                label_train_21_var.append(np.expand_dims(label_sample, axis=0))
                train_samples += 1
                print(f"Training Sample {train_samples}: {file_path} | HRRR: {hrrr_file.name}")

        except Exception as e:
            print(f"⚠️ Error on {file_path}: {e}")

print("\n================ FINAL ZARR STORE SUMMARY ================")
print(f"Training Input Zarr Shape:    {training_set_21_var.shape}")
print(f"Training Label Zarr Shape:    {label_train_21_var.shape}")
print(f"Validation Input Zarr Shape:  {validation_set_21_var.shape}")
print(f"Validation Label Zarr Shape:  {label_val_21_var.shape}")
print(f"Pipeline Completed:           {datetime.now().strftime('%H:%M:%S')}")
print("==========================================================")
