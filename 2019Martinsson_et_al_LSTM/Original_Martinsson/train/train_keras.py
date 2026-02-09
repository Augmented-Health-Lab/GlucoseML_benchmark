import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
import os

def train(model, x_train, y_train, x_valid, y_valid, batch_size, epochs,
        patience, shuffle, artifacts_path):
    print(tf.__version__)
    print("Sample x_train[0]:", x_train[0])
    print("Sample y_train[0]:", y_train[0])
    print("Shape x_train:", x_train.shape)
    print("Shape y_train:", y_train.shape)

    history = model.fit(
        x_train,
        y_train,
        validation_data = (x_valid, y_valid),
        epochs          = epochs,
        batch_size      = batch_size,
        shuffle         = shuffle,
        callbacks       = [
            tf.keras.callbacks.EarlyStopping(
                monitor  = 'val_loss',
                patience = patience,
                mode     = 'min'
            ),
            tf.keras.callbacks.TensorBoard(
                
                log_dir=artifacts_path,
                write_graph=False,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath       = os.path.join(artifacts_path, "model.keras"),
                monitor        = 'val_loss',
                save_best_only = True,
                save_freq      = 'epoch'
            )
        ]
    )
    return model
