
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Paths
# ----------------------------------
DATA_DIR = Path("split_data")
MODEL_PATH = Path("zat_results/best_zat.keras")
OUTPUT_DIR = Path("zat_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------------
# 2. Define custom Liquid layer
# ----------------------------------
@tf.keras.utils.register_keras_serializable()
class SimpleLiquidLayer(tf.keras.layers.Layer):

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


# ----------------------------------
# 3. Load test data
# ----------------------------------
print("Loading test data...")

with np.load(DATA_DIR / "test.npz") as data:
    X_test = data["X"].astype(np.float32)
    y_test = data["y"].astype(np.int32)

# ----------------------------------
# 4. Load saved model
# ----------------------------------
print("Loading trained ZAT-LNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"SimpleLiquidLayer": SimpleLiquidLayer},
    compile=False
)

# ----------------------------------
# 5. Predict test labels
# ----------------------------------
print("Predicting test data...")

probabilities = model.predict(
    X_test,
    batch_size=512,
    verbose=1
)

y_pred = np.argmax(probabilities, axis=1)

# ----------------------------------
# 6. Calculate evaluation metrics
# ----------------------------------
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n========== TEST RESULTS ==========")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Benign", "Malicious"],
    zero_division=0
))

# ----------------------------------
# 7. Confusion Matrix
# ----------------------------------
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malicious"]
)

display.plot(values_format="d")
plt.title("ZAT-LNN Confusion Matrix")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)
plt.show()

print("\nResults saved in zat_results folder.")
