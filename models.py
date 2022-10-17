import tensorflow as tf

def GRU_model():

    model = tf.keras.Sequential([
        tf.keras.layers.GRU(512, return_sequences=True,
                             input_shape = [None, max_id], dropout=0.2),
        tf.keras.layers.GRU(512, return_sequences=True,
                             dropout=0.2),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(max_id, activation="softmax"))
    ])

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam())

    return model
