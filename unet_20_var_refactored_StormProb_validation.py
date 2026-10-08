#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
"""
Developed with the assistance of the Gemini AI code assistant.
Author: Mamoudou Ba
Refactored by GEMINI: Fast U-Net 20-Feature HRRR Storm Probability Batch Inference Pipeline
                      Processes all hours (00-23 UTC) and lead times (f01-f08)
                      for March 1 - October 31 (2021 & 2022).
"""

# Disable C-level threading collisions & suppress TensorFlow GPU search on CPU nodes
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import netCDF4
import numpy as np
import pygrib

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# -----------------------------------------------------------------------------
# Configuration Constants & Paths
# -----------------------------------------------------------------------------
NX, NY = 501, 251
K_MINUS_C = 273.15
TWO_OVER_SEVEN = 0.2857142857142857

BASE_DIR = Path('/contrib/Mamoudou.Ba').resolve()
MODEL_FILE = BASE_DIR / 'storm_classification/unet_20_var_filters_80_wgt_0.1_patience_10_f01_f08.h5'
TEMPLATE_NC = BASE_DIR / 'storm_classification/template.nc'
OUTPUT_DIR = BASE_DIR / 'validation_data'
SCALER_SAVE_PATH = BASE_DIR / 'hrrr_scaler_20var.joblib'


# -----------------------------------------------------------------------------
# Custom Keras Layers & Objects
# -----------------------------------------------------------------------------
class BiasInit(Initializer):
    """Custom Bias Initializer matching training script."""
    def __call__(self, shape, dtype=None):
        values = np.zeros(shape, dtype=np.float32)
        values[...] = 0.01
        values[-1] = 1.0 - 0.01 * shape[-1]
        return K.constant(values, dtype=dtype, shape=shape)


def spatial_softmax(x):
    """Spatial softmax activation along channel axis."""
    return tf.keras.activations.softmax(x, axis=-1)


def softmaxND(x):
    """Calculate the softmax measure on the input tensor across arbitrary dimensions."""
    theShape = K.shape(x)
    firstDim = K.prod(theShape[0:-1])
    in2D = K.reshape(x, (firstDim, theShape[-1]))
    out2D = K.softmax(in2D)
    return K.reshape(out2D, theShape)


def weighted_categorical_crossentropy(weights):
    """Placeholder loss function required when loading compiled model."""
    weights_tensor = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        return -K.sum(y_true * K.log(y_pred) * weights_tensor, axis=-1)

    return loss


# -----------------------------------------------------------------------------
# Predictor Extraction Function (20 Features)
# -----------------------------------------------------------------------------
def extract_raw_20_predictors(file_path) -> np.ndarray:
    """Extracts raw 20 HRRR predictors into shape (NY, NX, 20)."""
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
        # Fallback: Message index position lookup
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

    # Derived Wind Speeds & Vertical Wind Shear Calculations
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

    raw_list_transposed = [arr.T if arr.shape == (NX, NY) else arr for arr in raw_list]
    predictors_3d = np.stack(raw_list_transposed, axis=-1).astype(np.float32)
    return np.nan_to_num(predictors_3d, nan=0.0, posinf=0.0, neginf=0.0)


# -----------------------------------------------------------------------------
# Main Batch Pipeline
# -----------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMPLATE_NC.exists():
        raise FileNotFoundError(f"Template NetCDF file not found: {TEMPLATE_NC}")
    if not SCALER_SAVE_PATH.exists():
        raise FileNotFoundError(f"Saved joblib StandardScaler file not found: {SCALER_SAVE_PATH}")
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"U-Net Model file not found: {MODEL_FILE}")

    # Load persistent components ONCE at startup
    print(f"Loading joblib StandardScaler: {SCALER_SAVE_PATH}")
    global_scaler = joblib.load(SCALER_SAVE_PATH)

    with netCDF4.Dataset(TEMPLATE_NC, 'r') as hrrrDataset:
        lat_vals = hrrrDataset.variables['latitude'][:]
        lon_vals = hrrrDataset.variables['longitude'][:]

    custom_objects = {
        'BiasInit': BiasInit,
        'biasInit': BiasInit,
        '<lambda>': spatial_softmax,
        'softmaxND': softmaxND,
        'loss': weighted_categorical_crossentropy(np.array([0.2, 1.0], dtype=np.float32))
    }

    print(f"Loading 20-Feature U-Net Model: {MODEL_FILE}")
    with custom_object_scope(custom_objects):
        model = load_model(MODEL_FILE, compile=False)

    print("🚀 U-Net model loaded successfully. Starting batch loop (2021-2022 | March 1 - Oct 31 | f01-f08)...\n")

    #years = [2021, 2022]
    years = [2021]

    for year in years:
        start_date = datetime(year, 3, 17)
        end_date = datetime(year, 3, 17)
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            daily_out_dir = OUTPUT_DIR / date_str
            daily_out_dir.mkdir(parents=True, exist_ok=True)

            for hr in range(24):
                cycle_str = f"{hr:02d}"

                # LOOP OVER LEAD TIMES f01 THROUGH f08
                for lead in range(1, 9):
                    fct_str = f"{lead:02d}"

                    # Input GRIB2 path checks
                    input_grib2 = BASE_DIR / f"hrrr_501_251/{date_str}/{date_str}_t{cycle_str}zprsf{fct_str}.grib2"
                    if not input_grib2.exists():
                       continue

                    # Standardized U-Net 20-var output path
                    pred_file = daily_out_dir / f"prob_{date_str}_unet_20_var_t{cycle_str}zf{fct_str}_wgt_0.1_f{fct_str}.nc"

                    # Skip if forecast file already exists
                    if pred_file.exists():
                        continue

                    # Calculate target valid forecast datetime
                    analysis_time = datetime.strptime(f"{date_str}{cycle_str}", "%Y%m%d%H")
                    valid_target_time = analysis_time + timedelta(hours=lead)
                    epoch_seconds = (valid_target_time - datetime(1970, 1, 1)).total_seconds()
                    valid_time_str = valid_target_time.strftime('%Y-%m-%d %H:%M:%S UTC')

                    try:
                        raw_predictors = extract_raw_20_predictors(input_grib2)
                    except Exception as e:
                        print(f"Error reading GRIB2 {input_grib2.name}: {e}")
                        continue

                    # Scale 20 predictor fields
                    scaled_flat = global_scaler.transform(raw_predictors.reshape(-1, 20))
                    scaled_predictors = scaled_flat.reshape(NY, NX, 20).astype(np.float32)
                    feature_sample_4d = np.expand_dims(scaled_predictors, axis=0)

                    # Fast U-Net inference
                    probs = model(feature_sample_4d, training=False).numpy()
                    
                    # Extract storm probability channel (channel 1) scaled to 0-100%
                    prob_grid = probs[0, ..., 1] * 100.0

                    # Write Output NetCDF File
                    with netCDF4.Dataset(pred_file, 'w', clobber=True) as predDataset:
                        predDataset.Conventions = "CF-1.8"
                        predDataset.institution = "NOAA/NWS/STI/MDL"
                        predDataset.title = f"1-hr Convection Probability Prediction (U-Net 20 Features) - Valid {valid_time_str}"

                        predDataset.createDimension('time', None)
                        predDataset.createDimension('latitude', NY)
                        predDataset.createDimension('longitude', NX)

                        timeVar = predDataset.createVariable('time', float, ('time',))
                        yVar = predDataset.createVariable('latitude', float, ('latitude',))
                        xVar = predDataset.createVariable('longitude', float, ('longitude',))

                        timeVar.standard_name = "time"
                        timeVar.long_name = "Valid Forecast Target Time"
                        timeVar.units = "seconds since 1970-01-01T00:00:00Z"
                        timeVar.axis = "T"
                        timeVar.comment = valid_time_str
                        timeVar[0] = epoch_seconds

                        yVar.standard_name = "latitude"
                        yVar.units = "degrees_north"
                        yVar.axis = "Y"
                        yVar[...] = lat_vals

                        xVar.standard_name = "longitude"
                        xVar.units = "degrees_east"
                        xVar.axis = "X"
                        xVar[...] = lon_vals

                        predVar = predDataset.createVariable('prob', float, ('time', 'latitude', 'longitude'), fill_value=-1.0)
                        predVar.valid_min = 0.0
                        predVar.valid_max = 100.0
                        predVar.standard_name = "1-hr Convection Prob %"
                        predVar.long_name = "1-hr Convection Probability in percent"
                        predVar[0, :, :] = prob_grid

                    print(f"✅ Generated 20-Var U-Net: Date={date_str} t{cycle_str}z f{fct_str} -> {pred_file.name}")

            current_date += timedelta(days=1)

    print("\n🎉 U-Net Batch processing complete across all dates and lead times!")


if __name__ == "__main__":
    main()
