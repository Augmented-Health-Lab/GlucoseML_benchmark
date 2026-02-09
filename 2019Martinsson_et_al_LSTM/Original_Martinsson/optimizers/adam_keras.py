from tensorflow.keras.optimizers.legacy import Adam

def load(cfg):
    learning_rate = float(cfg['learning_rate'])  # parse from config
    return Adam(learning_rate)
