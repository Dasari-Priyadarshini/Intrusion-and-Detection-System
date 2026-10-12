
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    LearningRateScheduler
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
DATA_DIR = Path("split_data")
TRAIN_FILE = DATA_DIR / "train.npz"

OUTPUT_DIR = Path("zat_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 512

np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------------------------
# 2. Load training, validation and test data
#    SMOTE is NOT used
# --------------------------------------------------
print("Loading original training data (without SMOTE)...")

with np.load(TRAIN_FILE) as d:
    X_train = d["X"].astype(np.float32)
    y_train = d["y"].astype(np.int32)

print("Loading validation data...")

with np.load(DATA_DIR / "validation.npz") as d:
    X_val = d["X"].astype(np.float32)
    y_val = d["y"].astype(np.int32)

print("Loading test data...")

with np.load(DATA_DIR / "test.npz") as d:
    X_test = d["X"].astype(np.float32)
    y_test = d["y"].astype(np.int32)

print("\nDataset shapes:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nTraining class distribution:")
print("Benign:", np.sum(y_train == 0))
print("Malicious:", np.sum(y_train == 1))

# --------------------------------------------------
# 3. Custom Liquid Neural Network layer
# --------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class SimpleLiquidLayer(layers.Layer):

    def __init__(self, units, tau_init=1.0, iterations=3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.tau_init = tau_init
        self.iterations = iterations

    def build(self, input_shape):
        input_dim = int(input_shape[-1])

        self.W_input = self.add_weight(
            name="W_input",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        self.W_recurrent = self.add_weight(
            name="W_recurrent",
            shape=(self.units, self.units),
            initializer="orthogonal",
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )

        initial_tau = np.log(
            np.expm1(max(self.tau_init - 0.1, 0.01))
        )

        self.tau = self.add_weight(
            name="tau",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(initial_tau),
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):
        h = tf.zeros(
            (tf.shape(inputs)[0], self.units),
            dtype=inputs.dtype
        )

        tau_eff = tf.nn.softplus(self.tau) + 0.1

        for _ in range(self.iterations):
            target = tf.tanh(
                tf.matmul(inputs, self.W_input)
                + tf.matmul(h, self.W_recurrent)
                + self.bias
            )

            h = h + (target - h) / tau_eff

        return h

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "tau_init": self.tau_init,
            "iterations": self.iterations
        })
        return config


# --------------------------------------------------
# 4. Build ZAT-style architecture
# --------------------------------------------------
inputs = layers.Input(
    shape=(X_train.shape[1],),
    name="network_features"
)

x = layers.Dense(256, activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

# Parallel liquid processing branches
liquid_fast = SimpleLiquidLayer(
    64, tau_init=0.5, name="liquid_layer_1"
)(x)

liquid_slow = SimpleLiquidLayer(
    32, tau_init=1.0, name="liquid_layer_2"
)(x)

liquid_fast = layers.Dropout(0.5)(liquid_fast)
liquid_slow = layers.Dropout(0.5)(liquid_slow)

combined = layers.Concatenate()(
    [liquid_fast, liquid_slow]
)

x = layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(combined)

x = layers.BatchNormalization()(x)
x = layers.Dropout(0.6)(x)

x = layers.Dense(
    64,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs, name="ZAT_LNN")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 5. Save the best model
# --------------------------------------------------
checkpoint = ModelCheckpoint(
    str(OUTPUT_DIR / "best_zat.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# --------------------------------------------------
# 6. Phase 1: Initial training
# --------------------------------------------------
print("\nPhase 1: Initial training")

def cosine_schedule(epoch):
    warmup_epochs = 5

    if epoch < warmup_epochs:
        return 0.001 * (epoch + 1) / warmup_epochs

    progress = (epoch - warmup_epochs) / (30 - warmup_epochs)

    return 0.0001 + 0.5 * (0.001 - 0.0001) * (
        1 + np.cos(np.pi * progress)
    )

phase1_callbacks = [
    LearningRateScheduler(cosine_schedule),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    ),
    checkpoint
]

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=BATCH_SIZE,
    callbacks=phase1_callbacks,
    shuffle=True
)

model.save(OUTPUT_DIR / "phase1_zat.keras")

# --------------------------------------------------
# 7. Phase 2: Fine-tuning
# --------------------------------------------------
print("\nPhase 2: Fine-tuning")

model.optimizer.learning_rate.assign(0.0001)

phase2_callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    checkpoint
]

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=BATCH_SIZE,
    callbacks=phase2_callbacks,
    shuffle=True
)

# --------------------------------------------------
# 8. Evaluate on untouched test data
# --------------------------------------------------
best_model = tf.keras.models.load_model(
    OUTPUT_DIR / "best_zat.keras"
)

print("\nEvaluating on untouched test set...")

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# --------------------------------------------------
# 9. Save final model
# --------------------------------------------------
best_model.save(OUTPUT_DIR / "final_zat.keras")

print("\nTraining and evaluation completed without SMOTE.")

