#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
# Author: Mamoudou Ba - December 2020 -
# Refactored by GEMINI: Single-Pass Validation Data Generator using Pre-Fitted StandardScaler
#
import numpy as np
import pygrib
from datetime import datetime
import dateutil.parser
import numpy.ma as ma
import netCDF4
import math
from pathlib import Path
import glob

from netCDF4 import Dataset
from tensorflow.keras.utils import to_categorical

import zarr
import os
import joblib
from sklearn.preprocessing import StandardScaler


def leap_year(y):
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    if y % 4 == 0:
        return True
    else:
        return False


nx, ny = (501, 251)
nxny = nx * ny

BASE_DIR = Path("/contrib/Mamoudou.Ba").resolve()
SCALER_LOAD_PATH = BASE_DIR / "cnn_scaler_13var.joblib"

# Load the pre-fitted StandardScaler trained on the training data
if not os.path.exists(SCALER_LOAD_PATH):
    raise FileNotFoundError(f"❌ Training scaler not found at: {SCALER_LOAD_PATH}. Please run the training script first.")

print(f"📦 Loading pre-fitted StandardScaler from: {SCALER_LOAD_PATH}")
global_scaler = joblib.load(SCALER_LOAD_PATH)

# Creating Zarr validation stores dynamically (starting at 0 samples)
zarr_input_name = 'zarr_input_validation_13_var_10km_f01_f08'
if os.path.exists(zarr_input_name):
    cmd = f"rm -rf {zarr_input_name}"
    os.system(cmd)

zarr_label_name = 'zarr_label_validation_13_var_10km_f01_f08'
if os.path.exists(zarr_label_name):
    cmd = f"rm -rf {zarr_label_name}"
    os.system(cmd)

training_set_13_var = zarr.open_array(zarr_input_name, mode='w', shape=(0, ny, nx, 13), chunks=(5, ny, nx, 13), dtype=np.float32)
label_13_var = zarr.open_array(zarr_label_name, mode='w', shape=(0, ny, nx, 2), chunks=(5, ny, nx, 2), dtype=np.float32)

number_of_day_in_a_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("****** Processing starts on date and time ", current_time)

lead_hour_configs = [
    (1, "01"),
    (2, "02"),
    (3, "03"),
    (4, "04"),
    (5, "05"),
    (6, "06"),
    (7, "07"),
    (8, "08"),
]

# =========================================================================
# SINGLE PASS: TRANSFORM 2025 VALIDATION DATA & WRITE TO ZARR
# =========================================================================
print("\n" + "=" * 70)
print("Normalizing 2025 Validation Grids using Training Scaler...")
print("=" * 70)

i = 0

for offset_hrs, lead_str in lead_hour_configs:
    print(f"--- Processing Lead Hour f{lead_str} ---")

    for file_path in glob.glob('mrms_501_251/20*/*.grib2'):
        url_str = file_path.split('/')
        date_str = url_str[1]
        path_str = url_str[2]
        split_file_name = path_str.split('_')
        time_str = split_file_name[3].split('-')

        hour_str = time_str[1][0:2]
        mrms_minutes = time_str[1][2:4]

        if int(mrms_minutes) > 15:
            continue
        month_str = date_str[4:6]
        day_str = date_str[6:8]
        year_str = date_str[0:4]

        # Year Filter: ONLY process 2025 validation data
        if year_str.strip() != '2025':
            continue

        imonth = int(month_str)
        iyear = int(year_str)
        day_analysis_time_int = int(day_str)
        date_analysis_time_str = date_str
        analysis_time_int = int(hour_str) - offset_hrs

        if (int(hour_str) - offset_hrs) < 0:
            if day_analysis_time_int == 1:
                day_analysis_time_int = number_of_day_in_a_month[imonth - 1]
                imonth = imonth - 1
                if imonth == 0:
                    imonth = 12
                    iyear = int(year_str) - 1
                    day_analysis_time_int = number_of_day_in_a_month[imonth - 1]
                year_analysis_time_str = str(iyear)
                month_analysis_time_str = f"{imonth:02d}"
                day_analysis_time_str = f"{day_analysis_time_int:02d}"
                date_analysis_time_str = year_analysis_time_str + month_analysis_time_str + day_analysis_time_str
            else:
                prev_day = int(day_str) - 1
                prev_day_str = f"{prev_day:02d}"
                date_analysis_time_str = year_str + month_str + prev_day_str
            analysis_time_str = f"{(24 + analysis_time_int):02d}"
        else:
            analysis_time_str = f"{analysis_time_int:02d}"
            date_analysis_time_str = date_str

        file_hrrr = f"/contrib/Mamoudou.Ba/hrrr_501_251_with_lightning/{date_analysis_time_str}/{date_analysis_time_str}_t{analysis_time_str}zprsf{lead_str}.grib2"
        f = f"{date_analysis_time_str}/{date_analysis_time_str}_t{analysis_time_str}zprs{lead_str}.grib2"

        if not os.path.exists(file_hrrr) or os.path.getsize(file_hrrr) < 2000000:
            continue
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 200000:
            continue

        try:
            mrms = pygrib.open(file_path)
            mrms_dbz = mrms.message(1).values[:, :]
            mrms.close()

            number_storm = np.count_nonzero((mrms_dbz >= 35.0) & (mrms_dbz < 95.0))
            if (number_storm / nxny) * 100.0 < 0.5:
                continue

            hrrr_fields = pygrib.open(file_hrrr)
            raw_vars = [
                hrrr_fields.message(15).values[:, :],  # PWAT
                hrrr_fields.message(8).values[:, :],   # MSLMA
                hrrr_fields.message(2).values[:, :],   # RH_850mb
                hrrr_fields.message(13).values[:, :],  # CAPE
                hrrr_fields.message(14).values[:, :],  # CIN
                hrrr_fields.message(12).values[:, :],  # LI
                hrrr_fields.message(10).values[:, :],  # U_surf
                hrrr_fields.message(11).values[:, :],  # V_surf
                hrrr_fields.message(4).values[:, :],   # U_850mb
                hrrr_fields.message(5).values[:, :],   # V_850mb
                hrrr_fields.message(6).values[:, :],   # Vvel_925mb
                hrrr_fields.message(1).values[:, :],   # Vvel_700mb
                hrrr_fields.message(7).values[:, :]    # hrrr dbz
            ]
            hrrr_fields.close()

            raw_3d = np.stack(raw_vars, axis=-1).astype(np.float32)

            # Transform using pre-fitted training StandardScaler
            scaled_flat = global_scaler.transform(raw_3d.reshape(-1, 13))
            scaled_3d = scaled_flat.reshape(ny, nx, 13).astype(np.float32)

            training_set_13_var.append(np.expand_dims(scaled_3d, axis=0))

            tstorm = np.where((mrms_dbz >= 35.0) & (mrms_dbz < 95.0), 1.0, 0.0)
            label_sample = to_categorical(tstorm, num_classes=2).astype(np.float32)
            label_13_var.append(np.expand_dims(label_sample, axis=0))

            print(i, file_path, f)
            i += 1

        except Exception as e:
            print(f"⚠️ Validation Processing Error on {file_path}: {e}")

icounter = i
print("Number of total hours for all lead times combined in validation set:", i, icounter)

current_time = now.strftime("%H:%M:%S")
print("End of processing =", current_time)
