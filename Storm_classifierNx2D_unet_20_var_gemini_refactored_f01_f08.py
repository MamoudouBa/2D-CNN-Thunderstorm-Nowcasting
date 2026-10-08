#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
"""
Author: Mamoudou Ba
Refactored: U-Net Convective Probability Pipeline (20 Feature Variables)
            Memory-optimized for GPU throughput with Loss Weight Sweep
"""

import gc
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import zarr
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Cropping2D,
    Input,
    MaxPool2D,
    Softmax,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model
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
INPUT_SHAPE = (251, 501, 20)  # 20 Predictor variables
EPOCHS = 100
BASE_FILTERS = 80             
LEARNING_RATE = 0.0001
PATIENCE = 10                  # Early Stopping Patience

# Loss weights loop array
#WEIGHTS_TO_TEST = [0.2]
WEIGHTS_TO_TEST = [0.05, 0.1, 0.2]

# Zarr store paths for 20-variable dataset
TRAIN_IN_PATH = '/contrib/Mamoudou.Ba/zarr_input_training_data_20_var_10km_f01_f08'
TRAIN_LBL_PATH = '/contrib/Mamoudou.Ba/zarr_label_training_data_20_var_10km_f01_f08'
VAL_IN_PATH = '/contrib/Mamoudou.Ba/zarr_input_validation_data_20_var_10km_f01_f08'
VAL_LBL_PATH = '/contrib/Mamoudou.Ba/zarr_label_validation_data_20_var_10km_f01_f08'


class ZarrDataSequence(tf.keras.utils.Sequence):
    """Memory-efficient, thread-safe Keras Sequence generator for chunked Zarr stores."""
    def __init__(self, in_zarr_path, lbl_zarr_path, shuffle=True):
        self.inputs = zarr.open_array(in_zarr_path, mode='r')
        self.labels = zarr.open_array(lbl_zarr_path, mode='r')
        self.batch_size = int(self.inputs.chunks[0])
        self.num_samples = int(self.inputs.shape[0])
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.shuffle = shuffle
        self.indices = np.arange(self.num_batches)
        self.on_epoch_end()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        batch_idx = self.indices[idx]
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)

        x = self.inputs[start_idx:end_idx, ...].astype(np.float32)
        y_raw = self.labels[start_idx:end_idx, ...]

        # Defensive target shape handling: Ensure 4D One-Hot targets (batch, 251, 501, 2)
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


def weighted_categorical_crossentropy(weights):
    """Weighted Categorical Crossentropy Loss."""
    weights_var = K.variable(weights)

    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss_val = y_true * K.log(y_pred) * weights_var
        return -K.sum(loss_val, axis=-1)

    return loss


def encoder_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = Activation('relu')(x)
    p = MaxPool2D(pool_size=(2, 2), strides=2)(x)
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = Concatenate()([x, skip_features])
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = Activation('relu')(x)
    return x


def build_unet(input_shape=INPUT_SHAPE, base_filters=BASE_FILTERS, num_classes=2):
    inputs = Input(input_shape)

    # Pad (251, 501) -> (256, 512) for exact power-of-2 pooling alignment
    x_padded = ZeroPadding2D(padding=((2, 3), (5, 6)))(inputs)

    # Contracting Path (Encoder)
    s1, p1 = encoder_block(x_padded, base_filters)
    s2, p2 = encoder_block(p1, base_filters * 2)
    s3, p3 = encoder_block(p2, base_filters * 4)
    s4, p4 = encoder_block(p3, base_filters * 8)

    # Bottleneck
    b1 = Conv2D(base_filters * 16, 3, padding='same')(p4)
    b1 = Activation('relu')(b1)
    b1 = Conv2D(base_filters * 16, 3, padding='same')(b1)
    b1 = Activation('relu')(b1)

    # Expansive Path (Decoder)
    s5 = decoder_block(b1, s4, base_filters * 8)
    s6 = decoder_block(s5, s3, base_filters * 4)
    s7 = decoder_block(s6, s2, base_filters * 2)
    s8 = decoder_block(s7, s1, base_filters)

    # Output Layer
    outputs_padded = Conv2D(num_classes, 1, padding='same')(s8)
    outputs_padded = Softmax(axis=-1)(outputs_padded)

    # Crop output back to (251, 501) to match ground truth target dimensions
    outputs = Cropping2D(cropping=((2, 3), (5, 6)))(outputs_padded)

    return Model(inputs=inputs, outputs=outputs, name=f'U-Net_Filters_{base_filters}')


def main():
    print(f"Beginning Pipeline Execution at {datetime.now().strftime('%H:%M:%S')}")

    # Ensure output target directory exists
    output_dir = Path('/contrib/Mamoudou.Ba/storm_classification')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Sequences once (re-used across weight iterations)
    train_seq = ZarrDataSequence(TRAIN_IN_PATH, TRAIN_LBL_PATH, shuffle=True)
    val_seq = ZarrDataSequence(VAL_IN_PATH, VAL_LBL_PATH, shuffle=False)

    print(f"Training samples: {train_seq.num_samples} ({len(train_seq)} batches)")
    print(f"Validation samples: {val_seq.num_samples} ({len(val_seq)} batches)")

    # Execute hyperparameter loop across loss weights
    for loss_weight in WEIGHTS_TO_TEST:
        print("\n" + "=" * 80)
        print(f"🚀 Training U-Net | Base Filters: {BASE_FILTERS} | Loss Weight: {loss_weight} | Patience: {PATIENCE}")
        print("=" * 80)

        weights_arr = np.array([loss_weight, 1.0], dtype=np.float32)

        model = build_unet(base_filters=BASE_FILTERS)
        model.compile(
            loss=weighted_categorical_crossentropy(weights_arr),
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=['accuracy']
        )

        model_save_path = str(output_dir / f'unet_20_var_filters_{BASE_FILTERS}_wgt_{loss_weight}_patience_{PATIENCE}_f01_f08.h5')
        history_save_path = str(output_dir / f'unet_20_var_filters_{BASE_FILTERS}_wgt_{loss_weight}_patience_{PATIENCE}_f01_f08.npy')

        # Callbacks setup per experiment iteration
        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=PATIENCE,
            verbose=1,
            mode="min",
            restore_best_weights=True
        )
        checkpoint = ModelCheckpoint(model_save_path, monitor="val_loss", save_best_only=True, verbose=1)
        cache_cleaner = ClearCache()

        callbacks = [early_stopping, checkpoint, cache_cleaner]

        # Train model iteration
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        # Save training metrics
        np.save(history_save_path, history.history)
        print(f"\nSuccessfully saved model to: {model_save_path}")
        print(f"Successfully saved history to: {history_save_path}")

        # Wiping GPU graphs and memory allocation before next weight iteration
        K.clear_session()
        del model
        gc.collect()

    print(f"\nPipeline Execution Complete! Finished at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
