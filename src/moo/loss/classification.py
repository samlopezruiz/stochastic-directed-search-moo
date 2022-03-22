import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import categorical_crossentropy
from keras.utils import losses_utils


def get_weight_different_to_ones(weight):
    if weight is None or all(v == 1. for v in weight):
        return None
    else:
        return tf.convert_to_tensor(weight, dtype=tf.float32)


class WeightedSCCE(keras.losses.Loss):

    def __init__(self, class_weight, from_logits=False, name='weighted_scce'):
        self.class_weight = get_weight_different_to_ones(class_weight)
        # self.reduction = keras.losses.Reduction.NONE
        self.reduction = losses_utils.ReductionV2.AUTO
        self.unreduced_scce = keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, name=name,
            reduction=self.reduction)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = self.unreduced_scce(y_true, y_pred, sample_weight)
        if self.class_weight is not None:
            weight_mask = tf.gather(self.class_weight, y_true)
            loss = tf.math.multiply(loss, weight_mask)
        return loss


class WeightedCategoricalCrossentropy:
    def __init__(self, fn_weights=None, fp_weights=None, from_logits=False, fp_mode=False):

        self.fn_weights = get_weight_different_to_ones(fn_weights)
        self.fp_weights = get_weight_different_to_ones(fp_weights)
        self.from_logits = from_logits
        self.fp_loss = fp_mode
        self.eps = tf.keras.backend.epsilon()

    def loss(self, y_true, y_pred):

        y_pred = tf.convert_to_tensor(y_pred)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        y_true = tf.cast(y_true, y_pred.dtype)

        if self.fp_loss:
            # consider only false classes
            y_true = tf.constant(1.) - y_true
            if self.fp_weights is not None:
                y_true = tf.multiply(self.fp_weights, y_true)

            y_pred = tf.constant(1.) - y_pred
        else:
            if self.fn_weights is not None:
                y_true = tf.multiply(self.fn_weights, y_true)

        y_pred = tf.where(tf.abs(y_pred) < self.eps, self.eps * tf.ones_like(y_pred), y_pred)
        return -tf.reduce_mean(tf.reduce_sum(tf.multiply(y_true, tf.math.log(y_pred)), axis=-1))
        # return (tf.reduce_sum(tf.multiply(y_true, tf.math.log(y_pred)), axis=-1))
        # have individual losses
        # return -tf.reduce_sum(tf.multiply(y_true, tf.math.log(y_pred)), axis=-1)


def moo_classification_weights(n_classes, individual_weight=2.):
    weights = np.ones((n_classes, n_classes))

    for i in range(n_classes):
        weights[i][i] = individual_weight

    return weights


if __name__ == "__main__":
    # %%

    w = tf.convert_to_tensor(np.ones(3), dtype=tf.float32)
    w2 = tf.ones_like(w)

    y_true = [[0, 1, 0], [0, 0, 1]]
    y_pred = [[0.05, 0.95, 0.0], [0.1, 0.8, 0.1]]
    loss = categorical_crossentropy(y_true, y_pred)
    print(loss.numpy())

    fn_crossentropy = WeightedCategoricalCrossentropy(fn_weights=[1., 1., 1.], fp_mode=False).loss
    fp_crossentropy = WeightedCategoricalCrossentropy(fp_weights=[1., 1., 1.], fp_mode=True).loss

    fn_loss = fn_crossentropy(y_true, y_pred).numpy()
    fp_loss = fp_crossentropy(y_true, y_pred).numpy()
    print(fn_loss)
    print(fp_loss)
