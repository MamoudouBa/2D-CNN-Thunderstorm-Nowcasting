#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
"""
Author: Mamoudou Ba
Refactored: 2D CNN Storm Classifier (21 Feature Variables)
            Optimized for maximum computational GPU throughput and memory stability.
"""

import gc
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import zarr
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Conv2D, Input, Softmax, SpatialDropout2D, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# TensorFlow log verbosity & memory growth setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpus = tf.config.list_physical_devices('GPU')
print(f"******** Num GPUs Available: {len(gpus)} *******")

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Configuration Constants
INPUT_SHAPE = (251, 501, 21)  # 21 Predictor variables
EPOCHS = 100
FILTER_SIZES = [80]
LEARNING_RATE = 0.0001
#LOSS_WEIGHT_VALS = [0.05, 0.1, 0.5]
LOSS_WEIGHT_VALS = [0.1]

# Zarr store paths for 21-variable dataset
TRAIN_IN_PATH  = '/contrib/Mamoudou.Ba/zarr_input_training_data_21_var_10km_f01_f08'
TRAIN_LBL_PATH = '/contrib/Mamoudou.Ba/zarr_label_training_data_21_var_10km_f01_f08'
VAL_IN_PATH    = '/contrib/Mamoudou.Ba/zarr_input_validation_data_21_var_10km_f01_f08'
VAL_LBL_PATH   = '/contrib/Mamoudou.Ba/zarr_label_validation_data_21_var_10km_f01_f08'

class ZarrDataSequence(tf.keras.utils.Sequence):
    """Memory-efficient, thread-safe Keras Sequence generator for chunked Zarr stores."""
    def __init__(self, in_zarr_path, lbl_zarr_path, shuffle=True):
        self.inputs = zarr.open_array(in_zarr_path, mode='r')
        self.labels = zarr.open_array(lbl_zarr_path, mode='r')
        self.batch_size = int(self.inputs.chunks[0])
        self.num_samples = int(self.inputs.shape[0])
        self.shuffle = shuffle
        self.indices = np.arange(self.__len__())
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx]
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        x = self.inputs[start_idx:end_idx, ...].astype(np.float32)
        y_raw = self.labels[start_idx:end_idx, ...]

        if y_raw.ndim == 3:
            y = tf.keras.utils.to_categorical(y_raw, num_classes=2).astype(np.float32)
        else:
            y = y_raw.astype(np.float32)

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
class ClearCache(Callback):
    """Garbage collection callback to clean GPU host RAM after each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


class BiasInit(Initializer):
    """Custom Bias Initializer for spatial categories."""
    def __call__(self, shape, dtype=None):
        values = np.zeros(shape, dtype=np.float32)
        values[...] = 0.01
        values[-1] = 1.0 - 0.01 * shape[-1]
        return K.constant(values, dtype=dtype, shape=shape)


def weighted_categorical_crossentropy(weights):
    """Weighted Categorical Crossentropy Loss."""
    weights_var = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss_val = y_true * K.log(y_pred) * weights_var
        return -K.sum(loss_val, axis=-1)

    return loss


def build_model(num_filters, input_shape=INPUT_SHAPE, filter_size=5, num_layers=5, dropout=0.20):
    """Builds 2D CNN with specified filter size and depth."""
    model = Sequential()
    model.add(Input(shape=input_shape))

    pad_size = filter_size // 2
    model.add(ZeroPadding2D(padding=(pad_size, pad_size), data_format='channels_last'))

    for _ in range(num_layers - 1):
        model.add(Conv2D(filters=num_filters, kernel_size=filter_size, activation='relu', data_format='channels_last'))
        model.add(ZeroPadding2D(padding=(pad_size, pad_size), data_format='channels_last'))
        if dropout > 0.0:
            model.add(SpatialDropout2D(dropout, data_format='channels_last'))

    # Conv layer per class
    model.add(Conv2D(filters=2, kernel_size=filter_size, activation='relu', data_format='channels_last'))

    # 1x1 Conv with Softmax along spatial channel dimension
    model.add(Conv2D(
        filters=2,
        kernel_size=1,
        data_format='channels_last',
        bias_initializer=BiasInit()
    ))
    model.add(Softmax(axis=-1))

    return model


def main():
    print(f"Beginning Pipeline Execution at {datetime.now().strftime('%H:%M:%S')}")

    # Ensure output target directory exists
    output_dir = Path('/contrib/Mamoudou.Ba/storm_classification')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Sequences
    train_seq = ZarrDataSequence(TRAIN_IN_PATH, TRAIN_LBL_PATH, shuffle=True)
    val_seq = ZarrDataSequence(VAL_IN_PATH, VAL_LBL_PATH, shuffle=False)

    print(f"Training samples: {train_seq.num_samples} ({len(train_seq)} batches)")
    print(f"Validation samples: {val_seq.num_samples} ({len(val_seq)} batches)")

    for num_filters in FILTER_SIZES:
        for wgt in LOSS_WEIGHT_VALS:
            print("\n" + "=" * 80)
            print(f"🚀 Starting Model Training with {num_filters} Filters (Loss Weight: {wgt})")
            print("=" * 80)

            weights_arr = np.array([wgt, 1.0], dtype=np.float32)

            model = build_model(num_filters=num_filters)
            model.compile(
                loss=weighted_categorical_crossentropy(weights_arr),
                optimizer=Adam(learning_rate=LEARNING_RATE),
                metrics=['accuracy']
            )

            model.summary()

            model_save_path = str(output_dir / f'cnn_std_scaler_21_var_{num_filters}_filters_wgt_{wgt}_f01_f08.h5')
            history_save_path = str(output_dir / f'cnn_std_scaler_21_var_{num_filters}_filters_wgt_{wgt}_f01_f08.npy')

            # Integrated Callbacks Pipeline
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    mode='min',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6,
                    verbose=1,
                ),
                ModelCheckpoint(
                    model_save_path,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    verbose=1,
                ),
                ClearCache(),
            ]

            # Train model
            history = model.fit(
                train_seq,
                validation_data=val_seq,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )

            # Save training metrics
            np.save(history_save_path, history.history)
            print(f"Successfully saved model to: {model_save_path}")
            print(f"Successfully saved history to: {history_save_path}")

            # Clear backend sessions to prevent memory accumulation
            K.clear_session()
            gc.collect()

    print(f"\nAll filter iterations complete! Finished at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
