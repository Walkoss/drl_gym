import numpy as np
import tensorflow.python.keras.backend as K
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import linear, tanh
from tensorflow.python.keras.layers import Dense, Input, Lambda
from tensorflow.python.keras.losses import mse
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam


def softmax_with_mask(tensor_and_mask):
    input_tensor, mask_tensor = tensor_and_mask
    min_tensor = K.min(input_tensor, axis=1, keepdims=True)
    positive_tensor = (min_tensor - input_tensor) * mask_tensor
    max_tensor = K.max(positive_tensor, axis=1, keepdims=True)
    exp_tensor = K.exp(positive_tensor - max_tensor)
    masked_tensor = exp_tensor * mask_tensor
    summed_tensor = K.sum(masked_tensor, axis=1, keepdims=True)
    return masked_tensor / (summed_tensor + 1e-10)


def build_ppo_loss(advantages, old_pred):
    def ppo_loss(y_true, y_pred):
        eps = 0.2
        entropy_loss = 0.001 * K.mean(
            K.sum(y_pred * K.log(y_pred + 1e-10), axis=1, keepdims=True)
        )  # Danger : le masque des actions possibles n'est pas pris en compte !!!
        r = y_pred * y_true / (old_pred * y_true + 1e-10)
        policy_loss = -K.mean(
            K.minimum(r * advantages, K.clip(r, 1 - eps, 1 + eps) * advantages)
        )
        return policy_loss + entropy_loss

    return ppo_loss


class PPOPolicyBrain:
    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        learning_rate: float = 0.0001,
        hidden_layers_count: int = 0,
        neurons_per_hidden_layer: int = 0,
    ):
        state_tensor = Input((state_dim,), name="state")
        mask_tensor = Input((output_dim,), name="mask")
        advantages_tensor = Input((1,), name="advantages")
        old_policy_tensor = Input((output_dim,), name="old_policy")

        hidden_tensor = state_tensor
        for i in range(hidden_layers_count):
            hidden_tensor = Dense(neurons_per_hidden_layer, activation=tanh)(
                hidden_tensor
            )

        hidden_tensor = Dense(output_dim, activation=linear)(hidden_tensor)
        policy_tensor = Lambda(lambda t: softmax_with_mask(t))(
            (hidden_tensor, mask_tensor)
        )

        self.model = Model(
            [state_tensor, mask_tensor, advantages_tensor, old_policy_tensor],
            [policy_tensor],
        )

        #print(self.model.summary())

        loss = build_ppo_loss(advantages_tensor, old_policy_tensor)

        self.model.compile(loss=loss, optimizer=Adam(lr=learning_rate))

    def predict(self, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.model.predict(
            (
                np.array((state,)),
                np.array((mask,)),
                np.zeros((1, 1)),
                np.ones((1, mask.shape[0])),
            )
        )[0]

    def train(
        self,
        states: np.ndarray,
        masks: np.ndarray,
        chosen_action_masks: np.ndarray,
        advantages: np.ndarray,
    ):
        old_predictions = self.model.predict(
            (states, masks, np.zeros((states.shape[0], 1)), np.ones_like(masks),)
        )
        self.model.fit(
            (states, masks, advantages, old_predictions),
            (chosen_action_masks,),
            epochs=10,
            batch_size=64,
            verbose=0,
        )

    def save_model(self, filename: str):
        self.model.save(f"{filename}_actor.h5")


class PPOValueBrain:
    def __init__(
        self,
        learning_rate: float = 0.0001,
        hidden_layers_count: int = 0,
        neurons_per_hidden_layer: int = 0,
    ):
        self.model = Sequential()

        for i in range(hidden_layers_count):
            self.model.add(Dense(neurons_per_hidden_layer, activation=tanh))

        self.model.add(Dense(1, activation=linear, use_bias=True))
        self.model.compile(loss=mse, optimizer=Adam(lr=learning_rate))

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict(np.array((state,)))[0]

    def train(self, states: np.ndarray, targets: np.ndarray):
        self.model.train_on_batch(states, targets)

    def save_model(self, filename: str):
        self.model.save(f"{filename}_critic.h5")


if __name__ == "__main__":
    import tensorflow as tf

    tf.enable_eager_execution()

    x = np.zeros((4, 3))
    x[3, 0] = -999999999999999999999
    x[3, 1] = -999999999999999999999
    x[3, 2] = 0

    x_mask = np.ones((4, 3))
    x_mask[0, 1] = 0.0
    x_mask[1, 0] = 0.0
    x_mask[2, 2] = 0.0
    x_mask[2, 1] = 0.0
    x_mask[3, 2] = 0.0

    print(softmax_with_mask((x, x_mask)))
