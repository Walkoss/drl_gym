from tensorflow.keras import Sequential
from tensorflow.keras.activations import linear, tanh
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np


class AlphaQNetwork:
    def __init__(
        self,
        output_dim: int,
        learning_rate: float = 0.0001,
        hidden_layers_count: int = 0,
        neurons_per_hidden_layer: int = 0,
    ):
        self.model = Sequential()

        for i in range(hidden_layers_count):
            self.model.add(Dense(neurons_per_hidden_layer, activation=tanh))

        self.model.add(Dense(output_dim, activation=linear, use_bias=False))
        self.model.compile(loss=mse, optimizer=Adam(lr=learning_rate))

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict(np.array((state,)))[0]

    def train(
        self, states: np.ndarray, chosen_action_masks: np.ndarray, targets: np.ndarray
    ):
        target_vecs = chosen_action_masks * np.expand_dims(targets, -1) + (
            1 - chosen_action_masks
        ) * self.model.predict(states)
        self.model.fit(states, target_vecs, epochs=10, verbose=0)

    def save_model(self, filename: str):
        self.model.save(f"{filename}.h5")
