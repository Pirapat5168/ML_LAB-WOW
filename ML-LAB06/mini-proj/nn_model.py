

from tensorflow import keras
from tensorflow.keras import layers


# ----------------------------------------------------------------
# NN configurations ที่จะนำมาเปรียบเทียบกัน
# key   = ชื่อ config (ใช้ตั้งชื่อไฟล์ output)
# value = list ของจำนวน neuron ในแต่ละ hidden layer
# ----------------------------------------------------------------
NN_CONFIGS = {
    "Small_1Layer":   [16],
    "Medium_2Layer":  [64, 32],
    "Large_3Layer":   [128, 64, 32],
}

# จำนวน epoch ที่จะนำมาเปรียบเทียบกัน (ใช้ config เดียวกัน คือ Medium_2Layer)
EPOCH_OPTIONS = [10, 30, 60]


def build_model(input_dim: int, hidden_layers: list, activation: str = "relu",
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
    """
    สร้าง Neural Network สำหรับ binary classification (gender: 0/1)

    Parameters
    ----------
    input_dim     : จำนวน feature ขาเข้า (50 จาก PCA)
    hidden_layers : list เช่น [64, 32] หมายถึง 2 hidden layer
                    layer แรก 64 neuron, layer สอง 32 neuron
    activation    : activation function ของ hidden layer (default relu)
    dropout_rate  : ป้องกัน overfitting
    learning_rate : learning rate ของ Adam optimizer
    """
    model = keras.Sequential(name=f"NN_{'-'.join(str(n) for n in hidden_layers)}")
    model.add(keras.Input(shape=(input_dim,)))

    for i, n_units in enumerate(hidden_layers):
        model.add(layers.Dense(n_units, activation=activation, name=f"hidden_{i+1}"))
        model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

    # output layer: sigmoid สำหรับ binary classification (Male/Female)
    model.add(layers.Dense(1, activation="sigmoid", name="output"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_model(input_dim=50, hidden_layers=NN_CONFIGS["Medium_2Layer"])
    m.summary()
