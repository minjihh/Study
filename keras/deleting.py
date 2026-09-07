

import time
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights=True,

)

start_time = time.time()
hsit = model.fit(x_train, y_train,
                 epochs=1000,
                 batch_size=16,
                 verbose=1,
                 validation_data=(x_val, y_val),
                 callbakcs=[es])

end_time = time.time()