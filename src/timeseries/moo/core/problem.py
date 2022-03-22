import tensorflow as tf
import numpy as np
from src.models.d_search.core.problem import ContinuationProblem
from src.moo.nn.utils import batch_array, predict_from_batches
from src.timeseries.utils.moo import params_conversion_weights, reconstruct_weights, get_last_layer_weights


class NNProblem(ContinuationProblem):

    def __init__(self,
                 y_true,
                 x_data,
                 model,
                 eval_fs,
                 n_obj=None,
                 x_tol_for_hash=None,
                 constraints_limits=None):

        self.model = model
        self.y_true = y_true
        self.x_data = x_data
        self.original_weights = None
        self.last_layer = None
        self.pre_last_layer_output = None
        self.eval_fs = eval_fs
        self.n_var = None
        self.ind_weights_params = None

        self.initialize()
        super().__init__(n_var=self.n_var,
                         n_obj=len(eval_fs) if n_obj is None else n_obj,
                         x_tol_for_hash=x_tol_for_hash,
                         constraints_limits=constraints_limits,
                         f=self.eval_model,
                         df=self.grad_model,
                         xl=-np.ones(self.n_var),
                         xu=np.ones(self.n_var))

    def initialize(self):
        self.original_weights, self.last_layer = self.get_last_layer()
        self.pre_last_layer_output = self.get_pre_last_layer_output()
        self.original_x = self.weights_to_individuals(self.original_weights)
        self.original_fx = self.eval_model(self.original_x)
        self.n_var = self.original_x.shape[1]

    def weights_to_individuals(self, weights):
        individual, params = params_conversion_weights(weights)

        if self.ind_weights_params is None:
            self.ind_weights_params = params

        # shape of individual = (n_vars,)
        return individual

    def individuals_to_weights(self, individual):
        return reconstruct_weights(individual, self.ind_weights_params)

    def get_pre_last_layer_output(self):
        return None

    def get_last_layer(self):
        # return weights and last layer
        return None, None

    def set_weights_last_layer(self, weights):
        self.last_layer.set_weights(weights)

    def predict(self):
        return self.last_layer(self.pre_last_layer_output)

    def eval_model(self, individual):
        if len(np.shape(individual)) == 1:
            individual = np.reshape(individual, (1, -1))

        weights = self.individuals_to_weights(individual)
        self.set_weights_last_layer(weights)
        losses = self._get_losses()

        return np.array(losses)

    def _get_losses(self):
        y_pred = self.predict()
        losses = [f(self.y_true, y_pred) for f in self.eval_fs]
        return losses

    def grad_model(self, individual):
        pass


class TFProblem(NNProblem):

    def get_pre_last_layer_output(self):
        outputs, output_map, data_map = self.model.predict_all(self.x_data, batch_size=128)
        pre_last_layer_output = outputs['transformer_output']
        return pre_last_layer_output

    def get_last_layer(self):
        # return weights and last layer
        original_weights, last_layer = get_last_layer_weights(self.model)
        return original_weights, last_layer

    def grad_model(self, individual):
        if len(np.shape(individual)) == 1:
            individual = np.reshape(individual, (-1, 1))

        weights = self.individuals_to_weights(individual)
        self.set_weights_last_layer(weights)

        grads = self._grad_model()

        numpy_grads = [[w.numpy() for w in d] for d in grads]
        grads_reshaped = [self.weights_to_individuals(grad) for grad in numpy_grads]
        return np.vstack(grads_reshaped)

    def _grad_model(self):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.predict()
            losses = [f(self.y_true, y_pred) for f in self.eval_fs]

        grads = [tape.gradient(loss, self.last_layer.trainable_variables) for loss in losses]
        return grads


class TsQuantileProblem(TFProblem):

    def __init__(self,
                 y_true,
                 x_data,
                 model,
                 eval_fs,
                 quantile_ix,
                 n_obj,
                 x_tol_for_hash=None,
                 constraints_limits=None):

        self.quantile_ix = quantile_ix
        super().__init__(y_true=y_true,
                         x_data=x_data,
                         model=model,
                         n_obj=n_obj,
                         eval_fs=eval_fs,
                         x_tol_for_hash=x_tol_for_hash,
                         constraints_limits=constraints_limits)

    def _get_losses(self):
        y_pred = self.predict()
        loss = self.eval_fs[0](self.y_true, y_pred)
        return [l.numpy() for l in loss[self.quantile_ix]]
        # return losses

    def _grad_model(self):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.predict()
            loss = self.eval_fs[0](self.y_true, y_pred)

        grads = [tape.gradient(loss[self.quantile_ix][i], self.last_layer.trainable_variables) for i in [0, 1]]
        return grads
