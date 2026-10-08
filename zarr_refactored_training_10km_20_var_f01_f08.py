#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
# Author: Mamoudou Ba - December 2020 -
# Refactored by GEMINI: Two-Pass StandardScaler with joblib Export & Dynamic Zarr (20 Features)
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

K_MINUS_C = 273.15


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
SCALER_SAVE_PATH = BASE_DIR / "cnn_scaler_20var.joblib"

# Creating Zarr file dynamically (starting at 0 samples)
zarr_file_name = 'zarr_input_training_data_20_var_10km_f01_f08'
if os.path.exists(zarr_file_name):
    cmd = f"rm -rf {zarr_file_name}"
    os.system(cmd)
zarr_file_name = 'zarr_label_training_data_20_var_10km_f01_f08'
if os.path.exists(zarr_file_name):
    cmd = f"rm -rf {zarr_file_name}"
    os.system(cmd)

training_set_20_var = zarr.open_array('zarr_input_training_data_20_var_10km_f01_f08', mode='w', shape=(0, ny, nx, 20), chunks=(5, ny, nx, 20), dtype=np.float32)
label_20_var = zarr.open_array('zarr_label_training_data_20_var_10km_f01_f08', mode='w', shape=(0, ny, nx, 2), chunks=(5, ny, nx, 2), dtype=np.float32)

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


def extract_raw_20_predictors(file_path) -> np.ndarray:
    """Extracts raw 20 HRRR predictors into shape (NY, NX, 20) without sign inversion."""
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
        vshear_surf_800, vshear_600_800
    ]

    raw_list_transposed = [arr.T if arr.shape == (nx, ny) else arr for arr in raw_list]
    predictors_3d = np.stack(raw_list_transposed, axis=-1).astype(np.float32)
    return np.nan_to_num(predictors_3d, nan=0.0, posinf=0.0, neginf=0.0)


# =========================================================================
# PASS 1: FIT STANDARD SCALER ON RAW PREDICTORS & SAVE TO JOBLIB
# =========================================================================
print("\n" + "=" * 70)
print("PASS 1: Fitting StandardScaler across lead hours f01-f08...")
print("=" * 70)

global_scaler = StandardScaler()
fit_count = 0

for offset_hrs, lead_str in lead_hour_configs:
    print(f"--- [Pass 1] Processing Lead Hour f{lead_str} ---")

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

        # Year Filters
        if int(year_str) == 2021 or int(year_str) == 2022 or int(year_str) > 2024:
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

        file_hrrr = f"/contrib/Mamoudou.Ba/hrrr_501_251/{date_analysis_time_str}/{date_analysis_time_str}_t{analysis_time_str}zprsf{lead_str}.grib2"

        if not os.path.exists(file_hrrr) or os.path.getsize(file_hrrr) < 2000000:
            continue
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 200000:
            continue

        print(file_path, file_hrrr)
        try:
            mrms = pygrib.open(file_path)
            mrms_dbz = mrms.message(1).values[:, :]
            mrms.close()

            number_storm = np.count_nonzero((mrms_dbz >= 35.0) & (mrms_dbz < 95.0))
            if (number_storm / nxny) * 100.0 < 0.5:
                continue

            raw_3d = extract_raw_20_predictors(file_hrrr)
            global_scaler.partial_fit(raw_3d.reshape(-1, 20))
            fit_count += 1

        except Exception as e:
            print(f"⚠️ Pass 1 Warning on {file_path}: {e}")

print(f"✅ Scaler fit complete on {fit_count} total grids.")
joblib.dump(global_scaler, SCALER_SAVE_PATH)
print(f"✅ Exported StandardScaler to: {SCALER_SAVE_PATH}")


# =========================================================================
# PASS 2: TRANSFORM WITH FITTED SCALER & WRITE DYNAMIC ZARR
# =========================================================================
print("\n" + "=" * 70)
print("PASS 2: Normalizing grids and writing to dynamic Zarr stores...")
print("=" * 70)

i = 0

for offset_hrs, lead_str in lead_hour_configs:
    print(f"--- [Pass 2] Processing Lead Hour f{lead_str} ---")

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

        # Year Filters
        if int(year_str) == 2021 or int(year_str) == 2022 or int(year_str) > 2024:
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

        file_hrrr = f"/contrib/Mamoudou.Ba/hrrr_501_251/{date_analysis_time_str}/{date_analysis_time_str}_t{analysis_time_str}zprsf{lead_str}.grib2"
        f = f"{date_analysis_time_str}/{date_analysis_time_str}_t{analysis_time_str}zprsf{lead_str}.grib2"

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

            raw_3d = extract_raw_20_predictors(file_hrrr)

            # Transform using loaded StandardScaler fit in Pass 1
            scaled_flat = global_scaler.transform(raw_3d.reshape(-1, 20))
            scaled_3d = scaled_flat.reshape(ny, nx, 20).astype(np.float32)

            training_set_20_var.append(np.expand_dims(scaled_3d, axis=0))

            tstorm = np.where((mrms_dbz >= 35.0) & (mrms_dbz < 95.0), 1.0, 0.0)
            label_sample = to_categorical(tstorm, num_classes=2).astype(np.float32)
            label_20_var.append(np.expand_dims(label_sample, axis=0))

            print(i, file_path, f)
            i += 1

        except Exception as e:
            print(f"⚠️ Pass 2 Error on {file_path}: {e}")

icounter = i
print("number of total hours for all lead time combined in the training is; ", i, icounter)

current_time = now.strftime("%H:%M:%S")
print("End of processing = ", current_time)
