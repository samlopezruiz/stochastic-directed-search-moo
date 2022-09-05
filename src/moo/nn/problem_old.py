from collections import defaultdict

import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from keras.utils.layer_utils import count_params
from src.moo.core.problem import ContinuationProblem
from src.moo.nn.utils import reconstruct_weights, params_conversion_weights, get_last_layer_weights, batch_array, \
    predict_from_batches, split_model, get_one_output_model


# class NNProblemOld(ContinuationProblem):
#
#     def __init__(self,
#                  y_train,
#                  x_train,
#                  model,
#                  eval_fs,
#                  y_valid=None,
#                  x_valid=None,
#                  metrics=None,
#                  n_obj=None,
#                  x_tol_for_hash=None,
#                  constraints_limits=None,
#                  batch_size=None):
#
#         self.model = model
#         self.y_train = y_train
#         self.batch_size = batch_size
#         self.y_valid = y_valid
#         self.x_train = x_train
#         self.x_valid = x_valid
#         self.original_weights = None
#         self.metrics = metrics
#         self.last_layer = None
#         self.last_layer_input_train = None
#         self.last_layer_input_valid = None
#         self.eval_fs = eval_fs
#         self.n_var = None
#         self.ind_weights_params = None
#
#         self.initialize()
#         super().__init__(n_var=self.n_var,
#                          n_obj=len(eval_fs) if n_obj is None else n_obj,
#                          x_tol_for_hash=x_tol_for_hash,
#                          constraints_limits=constraints_limits,
#                          f=self.eval_train,
#                          df=self.grad_model,
#                          xl=-np.ones(self.n_var),
#                          xu=np.ones(self.n_var))
#
#     def initialize(self):
#         print('...initializing problem')
#         self.original_weights, self.last_layer = self.get_last_layer()
#         self.last_layer_input_train = self.get_last_layer_input(self.x_train)
#         if self.x_valid is not None:
#             self.last_layer_input_valid = self.get_last_layer_input(self.x_valid)
#         self.original_x = self.weights_to_individuals(self.original_weights)
#         self.original_fx = self.eval_train(self.original_x)
#         self.n_var = self.original_x.shape[1]
#
#     def weights_to_individuals(self, weights):
#         individual, params = params_conversion_weights(weights)
#
#         if self.ind_weights_params is None:
#             self.ind_weights_params = params
#
#         # shape of individual = (n_vars,)
#         return individual
#
#     def individuals_to_weights(self, individual):
#         return reconstruct_weights(individual, self.ind_weights_params)
#
#     def get_last_layer(self):
#         # return weights and last layer
#         original_weights, last_layer = get_last_layer_weights(self.model)
#         return original_weights, last_layer
#
#     def get_last_layer_input(self, x):
#         aux_model = tf.keras.Model(inputs=self.model.inputs,
#                                    outputs=self.model.outputs + [self.last_layer.input])
#
#         y_pred, moo_layer_input = aux_model.predict(x)
#         return moo_layer_input
#
#     def set_weights_last_layer(self, weights):
#         self.last_layer.set_weights(weights)
#
#     def predict(self, x):
#         return self.last_layer(x)
#
#     def eval_train(self, individual):
#         return self.eval(individual, self.last_layer_input_train, self.y_train)
#
#     def eval_valid(self, individual):
#         if self.last_layer_input_valid is not None:
#             return self.eval(individual, self.last_layer_input_valid, self.y_valid)
#         else:
#             raise ValueError('x_valid/y_valid not provided')
#
#     def eval(self, individual, x, y):
#         if len(np.shape(individual)) == 1:
#             individual = np.reshape(individual, (1, -1))
#
#         weights = self.individuals_to_weights(individual)
#         self.set_weights_last_layer(weights)
#         losses = self._get_losses(x, y)
#
#         return np.array(losses)
#
#     def eval_with_metrics(self, ps, **kwargs):
#         metrics = []
#         for x in ps:
#             weights = self.individuals_to_weights(x)
#             self.set_weights_last_layer(weights)
#             y_pred = self.predict()
#             metrics.append(self._eval_metrics(self.y_train, y_pred, **kwargs))
#
#         return np.array(metrics)
#
#     def _get_losses(self, x, y):
#         y_pred = self.predict(x)
#         losses = [f(y, y_pred) for f in self.eval_fs]
#         return losses
#
#     def grad_model(self, individual):
#         raise NotImplementedError
#
#     def _eval_metrics(self, y_true, y_pred, **kwargs):
#         raise NotImplementedError


class NNProblem(ContinuationProblem):

    def __init__(self,
                 y_train,
                 x_train,
                 base_model,
                 moo_model,
                 eval_fs,
                 y_valid=None,
                 x_valid=None,
                 metrics=None,
                 n_obj=None,
                 x_tol_for_hash=None,
                 constraints_limits=None,
                 base_batch_size=None,
                 moo_batch_size=None,
                 moo_from_batches=False):

        self.moo_batch_size = moo_batch_size
        self.moo_from_batches = moo_from_batches
        self.base_model = base_model
        self.moo_model = moo_model
        self.y_train = y_train
        self.pred_batch_size = base_batch_size
        self.y_valid = y_valid
        self.x_train = x_train
        self.x_valid = x_valid
        self.original_weights = None
        self.metrics = metrics
        self.last_layer = None
        self.moo_model_input_train = None
        self.moo_model_input_valid = None
        self.x_train_batches = None
        self.x_valid_batches = None
        self.y_train_batches = None
        self.eval_fs = eval_fs
        self.n_var = None
        self.ind_weights_params = None
        self.weights_lbls = []

        self.initialize()
        super().__init__(n_var=self.n_var,
                         n_obj=len(eval_fs) if n_obj is None else n_obj,
                         x_tol_for_hash=x_tol_for_hash,
                         constraints_limits=constraints_limits,
                         f=self.eval_train,
                         df=self.grad_model,
                         xl=-np.ones(self.n_var),
                         xu=np.ones(self.n_var))

    def batch_inputs(self):
        print("... batching data")
        self.x_train_batches = batch_array(self.x_train, batch_size=self.pred_batch_size)
        if self.x_valid is not None:
            self.x_valid_batches = batch_array(self.x_valid, batch_size=self.pred_batch_size)

        self.y_train_batches = batch_array(self.y_train, batch_size=self.pred_batch_size)

    def get_moo_model_inputs(self):
        print("... getting moo model input")
        self.moo_model_input_train = predict_from_batches(self.base_model, self.x_train_batches,
                                                          to_numpy=False,
                                                          concat_output=False)
        if self.x_valid is not None:
            self.moo_model_input_valid = predict_from_batches(self.base_model, self.x_valid_batches,
                                                              to_numpy=False,
                                                              concat_output=False)

    def initialize(self):
        print('initializing problem')
        self.batch_inputs()
        self.get_moo_model_inputs()
        self.get_model_weights()

        self.original_x = self.weights_to_individuals(self.original_weights)
        self.original_fx = self.eval_train(self.original_x)
        self.n_var = self.original_x.shape[1]

    def weights_to_individuals(self, weights):
        individual, params = params_conversion_weights(weights)

        if self.ind_weights_params is None:
            self.ind_weights_params = params

        return individual

    def individuals_to_weights(self, individual):
        return reconstruct_weights(individual, self.ind_weights_params)

    def get_model_weights(self):
        weights = self.moo_model.trainable_weights
        # weight labels contain only the layer they correspond
        # kernel and bias labels are not included
        self.original_weights, self.weights_lbls = [w.numpy() for w in weights], [w.name.split('/')[0] for w in weights]

        # # return weights and last layer
        # original_weights, last_layer = get_last_layer_weights(self.base_model)
        # return original_weights, last_layer

    def get_moo_model_input(self, x):
        return self.base_model(x)

    def set_weights_moo_model(self, weights):
        # self.last_layer.set_weights(weights)
        weights_dict = defaultdict(list)

        # weights are represented as lists with the layer's name as key
        for lbl, w in zip(self.weights_lbls, weights):
            weights_dict[lbl].append(w)

        # replace weights of layers with trainable weights
        for layer in self.moo_model.layers:
            if layer.name in weights_dict:
                layer.set_weights(weights_dict[layer.name])

    def predict(self, x, convert_to_numpy=True):
        return predict_from_batches(self.moo_model, x,
                                    to_numpy=convert_to_numpy,
                                    concat_output=True)
        # return self.last_layer(x)

    def eval_train(self, individual):
        return self.eval(individual, self.moo_model_input_train, self.y_train)

    def eval_valid(self, individual):
        if self.moo_model_input_valid is not None:
            return self.eval(individual, self.moo_model_input_valid, self.y_valid)
        else:
            raise ValueError('x_valid/y_valid not provided')

    def eval(self, individual, x, y):
        if len(np.shape(individual)) == 1:
            individual = np.reshape(individual, (1, -1))

        weights = self.individuals_to_weights(individual)
        self.set_weights_moo_model(weights)
        losses = self._get_losses(x, y)

        return np.array(losses)

    # def eval_with_metrics(self, x, ps, **kwargs):
    #     metrics = []
    #     for x in ps:
    #         weights = self.individuals_to_weights(x)
    #         self.set_weights_moo_model(weights)
    #         y_pred = self.predict()x
    #         metrics.append(self._eval_metrics(self.y_train, y_pred, **kwargs))
    #
    #     return np.array(metrics)

    def _get_losses(self, x, y):
        y_pred = self.predict(x, convert_to_numpy=False)
        losses = [f(y, y_pred) for f in self.eval_fs]
        return losses

    def grad_model(self, individual):
        raise NotImplementedError

    def _eval_metrics(self, y_true, y_pred, **kwargs):
        raise NotImplementedError


class TFProblem(NNProblem):

    def grad_model(self, individual):
        if len(np.shape(individual)) == 1:
            individual = np.reshape(individual, (-1, 1))

        weights = self.individuals_to_weights(individual)
        self.set_weights_moo_model(weights)

        # grads are calculated always with train data
        return self.batch_gradients(self.moo_model_input_train, self.y_train_batches)

    def batch_gradients(self, x_batches, y_batches):
        grad_batches = []
        for x, y in zip(x_batches, y_batches):
            grads = self._batch_gradient(x, y)

            # convert standard shape of Jacobian
            numpy_grads = [[w.numpy() for w in d] for d in grads]
            grads_reshaped = [self.weights_to_individuals(grad) for grad in numpy_grads]
            grad_batches.append(np.vstack(grads_reshaped))

        # return mean of batch gradients
        return np.mean(np.array(grad_batches), axis=0)

    def _batch_gradient(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            # [x] because 'predict' receives batches as input
            y_pred = self.moo_model(x)
            # y_pred = self.predict([x], convert_to_numpy=False)
            losses = [f(y, y_pred) for f in self.eval_fs]
        grads = [tape.gradient(loss, self.moo_model.trainable_variables) for loss in losses]
        return grads


class ClassificationProblem(TFProblem):
    def _eval_metrics(self, y_true, y_pred, **kwargs):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        precision, recall, fbeta, support = precision_recall_fscore_support(y_true, y_pred)
        results = {'precision': precision, 'recall': recall, 'fbeta': fbeta, 'support': support}
        return results[kwargs.get('key', 'fbeta')]


class TsQuantileProblem(TFProblem):

    def __init__(self,
                 y_train,
                 x_train,
                 model,
                 eval_fs,
                 quantile_ix,
                 n_obj,
                 x_valid=None,
                 y_valid=None,
                 x_tol_for_hash=None,
                 constraints_limits=None,
                 base_batch_size=None,
                 moo_batch_size=None,
                 moo_model_size='small',
                 moo_from_batches=False,
                 ):

        self.quantile_ix = quantile_ix

        if moo_model_size == 'medium':
            intermediate_layers = ['layer_normalization_36', 'time_distributed_144']
        elif moo_model_size == 'small':
            intermediate_layers = ['layer_normalization_40']
        else:
            raise NotImplementedError

        model = get_one_output_model(model, output_layer_name='td_quantiles')
        models = split_model(model, intermediate_layers)

        print(tabulate([(k, float(count_params(model.trainable_weights))) for k, model in models.items()],
                       headers=['model', 'trainable weights'],
                       floatfmt=(None, ",.0f"),
                       tablefmt='psql',
                       stralign="right"))

        super().__init__(y_train=y_train,
                         x_train=x_train,
                         x_valid=x_valid,
                         y_valid=y_valid,
                         base_model=models['base_model'],
                         moo_model=models['trainable_model'],
                         n_obj=n_obj,
                         eval_fs=eval_fs,
                         base_batch_size=base_batch_size,
                         moo_batch_size=moo_batch_size,
                         x_tol_for_hash=x_tol_for_hash,
                         moo_from_batches=moo_from_batches,
                         constraints_limits=constraints_limits)

    def _get_losses(self, x, y):
        y_pred = self.predict(x, convert_to_numpy=False)
        loss = self.eval_fs[0](y, y_pred)
        return [l.numpy() for l in loss[self.quantile_ix]]

    def _batch_gradient(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            # [x] because 'predict' receives batches as input
            y_pred = self.moo_model(x)
            # y_pred = self.predict([x], convert_to_numpy=False)
            loss = self.eval_fs[0](y, y_pred)
        grads = [tape.gradient(loss[self.quantile_ix][i], self.moo_model.trainable_variables) for i in [0, 1]]
        return grads


class GradientTestsProblem(TsQuantileProblem):

    def get_moo_model_inputs(self):
        print("... getting moo model input")
        self.moo_model_input_train = predict_from_batches(self.base_model, self.x_train_batches,
                                                          to_numpy=False,
                                                          concat_output=True)

        self.moo_model_input_train_batched = predict_from_batches(self.base_model, self.x_train_batches,
                                                                  to_numpy=False,
                                                                  concat_output=False)

        if self.x_valid is not None:
            self.moo_model_input_valid = predict_from_batches(self.base_model, self.x_valid_batches,
                                                              to_numpy=False,
                                                              concat_output=True)

    def grad_model(self, individual):
        if len(np.shape(individual)) == 1:
            individual = np.reshape(individual, (-1, 1))

        weights = self.individuals_to_weights(individual)
        self.set_weights_moo_model(weights)

        if self.moo_from_batches:
            self.moo_model_input_train_batched = batch_array(self.moo_model_input_train, self.moo_batch_size)
            self.y_train_batches = batch_array(self.y_train, self.moo_batch_size)
            batched_grads = []
            for x, y in zip(self.moo_model_input_train_batched, self.y_train_batches):
                with tf.GradientTape(persistent=True) as tape:
                    # grads are calculated always with train data
                    y_pred = self.moo_model(x)
                    loss = self.eval_fs[0](y, y_pred)

                grads = [tape.gradient(loss[self.quantile_ix][i], self.moo_model.trainable_variables) for i in [0, 1]]

                weights_grads = [[w for w in d] for d in grads]
                grads_reshaped = [tf.concat([tf.reshape(w, [1, -1]) for w in grad], 1) for grad in weights_grads]
                batched_grads.append(tf.squeeze(tf.stack(grads_reshaped, axis=0)))

            batched_grads = tf.stack(batched_grads)
            return tf.reduce_mean(batched_grads, axis=0).numpy()

        else:
            with tf.GradientTape(persistent=True) as tape:
                # grads are calculated always with train data
                y_pred = self.moo_model(self.moo_model_input_train)
                loss = self.eval_fs[0](self.y_train, y_pred)

            grads = [tape.gradient(loss[self.quantile_ix][i], self.moo_model.trainable_variables) for i in [0, 1]]

            numpy_grads = [[w.numpy() for w in d] for d in grads]
            grads_reshaped = [self.weights_to_individuals(grad) for grad in numpy_grads]
            return np.vstack(grads_reshaped)

    def predict(self, x, convert_to_numpy=True):
        pred = self.moo_model(x)
        # return pred.numpy()
        return pred.numpy() if convert_to_numpy else pred
