"""
cnn_model.py
สร้าง / เทรน / บันทึก / ทำนาย ด้วย CNN model
รองรับการปรับ config: จำนวน convolutional block และจำนวน neuron ใน dense layer
เพื่อใช้เปรียบเทียบ configuration ต่างๆ ตามโจทย์ Lab
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(128, 128, 3), num_classes=2,
              conv_layers=3, base_filters=32, dense_neurons=128,
              dropout_rate=0.3):
    """
    สร้าง CNN โครงสร้าง: [Conv2D -> BatchNorm -> MaxPool] x conv_layers
                          -> Flatten -> Dense -> Dropout -> Dense(output)
    - conv_layers: จำนวน convolutional block (ยิ่งมาก โมเดลยิ่งลึก)
    - base_filters: จำนวน filter ของ block แรก (double ทุก block ถัดไป)
    - dense_neurons: จำนวน neuron ใน fully-connected layer
    """
    model = keras.Sequential(name=f"cnn_conv{conv_layers}_dense{dense_neurons}")
    model.add(layers.Input(shape=input_shape))

    filters = base_filters
    for _ in range(conv_layers):
        model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        filters *= 2  # double จำนวน filter ทุก block

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_neurons, activation="relu"))
    model.add(layers.Dropout(dropout_rate))

    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def train_cnn(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64,
              verbose=1, early_stopping_patience=3):
    """
    เทรนโมเดล คืน History object (เก็บ accuracy/loss ทุก epoch)
    ใช้ EarlyStopping: หยุดเทรนอัตโนมัติถ้า val_accuracy ไม่ดีขึ้นติดต่อกัน
    `early_stopping_patience` epoch แล้วดึง weight ที่ดีที่สุดกลับมาใช้เสมอ
    (restore_best_weights=True) จึงไม่กระทบคุณภาพโมเดล แค่ประหยัดเวลาเทรน
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[early_stop],
    )
    return history


def save_model(model, path):
    model.save(path)
    print(f"[cnn_model] บันทึกโมเดลที่ {path}")


def load_model(path):
    return keras.models.load_model(path)


def predict(model, X):
    """คืน predicted class (0/1) และ probability ของแต่ละภาพ"""
    probs = model.predict(X, verbose=0)
    if probs.shape[-1] == 1:  # binary sigmoid output
        probs = probs.flatten()
        preds = (probs >= 0.5).astype(int)
    else:  # softmax (multi-class)
        preds = probs.argmax(axis=1)
    return preds, probs
