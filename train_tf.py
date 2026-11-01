
import argparse
import json

import tensorflow as tf

from psoriscan.config import MODEL_DIR, MODEL_FILES, REPORT_DIR, SEED
from psoriscan.data import load_arrays
from psoriscan.reporting import save_history_plot
from psoriscan.tf_model import build_cnn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    x_train, y_train = load_arrays("train")
    x_val, y_val = load_arrays("val")
    print(f"train={len(x_train)} val={len(x_val)}")

    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(len(x_train), seed=args.seed).batch(args.batch_size).prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_FILES["tf"]
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, shuffle=False, verbose=2)

    out_dir = REPORT_DIR / "tf"
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items() if k != "learning_rate"}
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    save_history_plot(hist, out_dir / "training_curves.png", "TensorFlow CNN")
    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    main()
