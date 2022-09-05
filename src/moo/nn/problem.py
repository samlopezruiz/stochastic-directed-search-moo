from collections import defaultdict
from keras.utils.layer_utils import count_params
import tensorflow as tf
import numpy as np
from tabulate import tabulate

from src.moo.core.problem import ContinuationProblem
from src.moo.nn.utils import reconstruct_weights, params_conversion_weights, batch_from_list_or_array, \
    predict_from_batches, get_one_output_model, split_model


class NNProblem(ContinuationProblem):

    def __init__(self,
                 y_train,
                 x_train,
                 base_model,
                 moo_model,
                 eval_fs,
                 y_valid=None,
                 x_valid=None,
                 x_test=None,
                 y_test=None,
                 metrics=None,
                 n_obj=None,
                 use_gpu=True,
                 x_tol_for_hash=None,
                 constraints_limits=None,
                 base_batch_size=None,
                 moo_batch_size=None):

        self.use_gpu = use_gpu
        self.base_batch_size = base_batch_size
        self.moo_batch_size = moo_batch_size
        self.base_model = base_model
        self.moo_model = moo_model
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test
        self.x_train = x_train
        self.x_valid = x_valid
        self.x_test = x_test
        self.original_weights = None
        self.metrics = metrics
        self.moo_model_input_train = None
        self.moo_model_input_valid = None
        self.moo_model_input_test = None
        self.moo_model_input_train_unbatched = None
        self.moo_model_input_valid_unbatched = None
        self.moo_model_input_test_unbatched = None
        self.x_train_batches = None
        self.x_valid_batches = None
        self.x_test_batches = None
        self.y_train_batches = None
        self.eval_fs = eval_fs
        self.n_var = None
        self.ind_weights_params = None
        self.weights_lbls = []
        self.train_batch_ix = 0
        self.valid_batch_ix = 0
        self.test_batch_ix = 0

        self.initialize()
        super().__init__(n_var=self.n_var,
                         n_obj=len(eval_fs) if n_obj is None else n_obj,
                         x_tol_for_hash=x_tol_for_hash,
                         constraints_limits=constraints_limits,
                         f=self.eval_train,
                         df=self.grad_model,
                         xl=-np.ones(self.n_var),
                         xu=np.ones(self.n_var))

    @property
    def train_n_batches(self):
        return len(self.moo_model_input_train)

    @property
    def valid_n_batches(self):
        return len(self.moo_model_input_valid)

    @property
    def test_n_batches(self):
        return len(self.moo_model_input_test)

    def batch_base_inputs(self):
        print("batching base model input data...", end="")
        self.x_train_batches = batch_from_list_or_array(self.x_train, batch_size=self.base_batch_size)
        if self.x_valid is not None:
            self.x_valid_batches = batch_from_list_or_array(self.x_valid, batch_size=self.base_batch_size)
        if self.x_test is not None:
            self.x_test_batches = batch_from_list_or_array(self.x_test, batch_size=self.base_batch_size)
        print("Done!")

    def batch_moo_inputs(self):
        print("batching moo model input data...", end="")
        self.moo_model_input_train = batch_from_list_or_array(self.moo_model_input_train_unbatched,
                                                              batch_size=self.moo_batch_size)
        if self.moo_model_input_valid_unbatched is not None:
            self.moo_model_input_valid = batch_from_list_or_array(self.moo_model_input_valid_unbatched,
                                                                  batch_size=self.moo_batch_size)
        if self.moo_model_input_test_unbatched is not None:
            self.moo_model_input_test = batch_from_list_or_array(self.moo_model_input_test_unbatched,
                                                                 batch_size=self.moo_batch_size)

        self.y_train_batches = batch_from_list_or_array(self.y_train, batch_size=self.moo_batch_size)
        print("Done!")

    def get_moo_model_inputs(self):
        print("getting moo model input...", end="")
        self.moo_model_input_train_unbatched = predict_from_batches(self.base_model, self.x_train_batches,
                                                                    to_numpy=False,
                                                                    concat_output=True,
                                                                    use_gpu=self.use_gpu)
        if self.x_valid is not None:
            self.moo_model_input_valid_unbatched = predict_from_batches(self.base_model, self.x_valid_batches,
                                                                        to_numpy=False,
                                                                        concat_output=True,
                                                                        use_gpu=self.use_gpu)

        if self.x_test is not None:
            self.moo_model_input_test_unbatched = predict_from_batches(self.base_model, self.x_test_batches,
                                                                       to_numpy=False,
                                                                       concat_output=True,
                                                                       use_gpu=self.use_gpu)
        print("Done!")

    def shuffle_train_data(self, random_state=None):
        if isinstance(self.moo_model_input_train_unbatched, list):
            indices = tf.range(start=0, limit=tf.shape(self.moo_model_input_train_unbatched[0])[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices, seed=random_state)
            self.moo_model_input_train_unbatched = [tf.gather(x, shuffled_indices) for x in
                                                    self.moo_model_input_train_unbatched]
        else:
            indices = tf.range(start=0, limit=tf.shape(self.moo_model_input_train_unbatched)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices, seed=random_state)
            self.moo_model_input_train_unbatched = tf.gather(self.moo_model_input_train_unbatched, shuffled_indices)

        self.y_train = self.y_train[shuffled_indices, Ellipsis]

        self.batch_moo_inputs()

    def initialize(self):
        print('initializing problem')
        self.batch_base_inputs()
        self.get_moo_model_inputs()
        self.batch_moo_inputs()
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

    def eval_train(self, individual):
        return self.eval(individual, self.moo_model_input_train, self.y_train)

    def eval_valid(self, individual):
        if self.moo_model_input_valid is not None:
            return self.eval(individual, self.moo_model_input_valid, self.y_valid)
        else:
            raise ValueError('x_valid/y_valid not provided')

    def eval_test(self, individual):
        if self.moo_model_input_test is not None:
            return self.eval(individual, self.moo_model_input_test, self.y_test)
        else:
            raise ValueError('x_test/y_test not provided')

    def eval_individuals(self, individuals, subset='train'):
        if subset == 'train':
            return np.array([self.eval_train(ind) for ind in individuals])
        elif subset == 'valid':
            return np.array([self.eval_valid(ind) for ind in individuals]) if self.x_valid is not None else None
        elif subset == 'test':
            return np.array([self.eval_test(ind) for ind in individuals]) if self.x_test is not None else None
        else:
            raise ValueError('subset must be train, valid or test')

    def eval(self, individual, x_batches, y):
        if len(np.shape(individual)) == 1:
            individual = np.reshape(individual, (1, -1))

        weights = self.individuals_to_weights(individual)
        self.set_weights_moo_model(weights)
        losses = self._get_losses(x_batches, y)

        return np.array(losses)

    def _get_losses(self, x_batches, y):
        y_pred = predict_from_batches(self.moo_model,
                                      x_batches,
                                      to_numpy=False,
                                      concat_output=True,
                                      use_gpu=self.use_gpu)

        losses = [f(y, y_pred) for f in self.eval_fs]
        return losses

    def grad_model(self, individual):
        raise NotImplementedError

    def grad_model_next_batch(self, individual):
        raise NotImplementedError

    def _eval_metrics(self, y_true, y_pred, **kwargs):
        raise NotImplementedError


class TFProblem(NNProblem):

    def grad_next_batch(self, individual):
        self.set_weights_from_individual(individual)

        grads = self._batch_gradient(self.moo_model_input_train[self.train_batch_ix],
                                     self.y_train_batches[self.train_batch_ix])
        grads = self.reshape_gradient(grads)

        self.n_grad_evals += 1 / len(self.moo_model_input_train)
        # increment batch index
        self.train_batch_ix = (self.train_batch_ix + 1) % len(self.moo_model_input_train)

        return grads

    def set_weights_from_individual(self, individual):
        if len(np.shape(individual)) == 1:
            individual = np.reshape(individual, (-1, 1))

        weights = self.individuals_to_weights(individual)
        self.set_weights_moo_model(weights)

    def grad_model(self, individual):
        self.set_weights_from_individual(individual)

        # grads are calculated always with train data
        return self.batch_gradients(self.moo_model_input_train, self.y_train_batches)

    def reshape_gradient(self, grads):
        with tf.device('/device:GPU:0' if self.use_gpu else "/cpu:0"):
            # convert standard shape of Jacobian
            weights_grads = [[w for w in d] for d in grads]
            grads_reshaped = [tf.concat([tf.reshape(w, [1, -1]) for w in grad], 1) for grad in weights_grads]
            ans = tf.squeeze(tf.stack(grads_reshaped, axis=0))
        return ans

    def batch_gradients(self, x_batches, y_batches):
        with tf.device('/device:GPU:0' if self.use_gpu else "/cpu:0"):
            grad_batches = []
            for x, y in zip(x_batches, y_batches):
                grads = self._batch_gradient(x, y)
                grad_batches.append(self.reshape_gradient(grads))

            # return mean of batch gradients
            grad_batches = tf.stack(grad_batches)
            ans = tf.reduce_mean(grad_batches, axis=0)
        return ans

    def _batch_gradient(self, x, y):
        with tf.device('/device:GPU:0' if self.use_gpu else "/cpu:0"):
            with tf.GradientTape(persistent=True) as tape:
                y_pred = self.moo_model(x)
                losses = [f(y, y_pred) for f in self.eval_fs]
            grads = [tape.gradient(loss, self.moo_model.trainable_variables) for loss in losses]
        return grads


class TFSplitProblem(TFProblem):

    def __init__(self,
                 y_train,
                 x_train,
                 model,
                 eval_fs,
                 n_obj,
                 intermediate_layers,
                 output_layer_name,
                 x_valid=None,
                 y_valid=None,
                 x_test=None,
                 y_test=None,
                 x_tol_for_hash=None,
                 constraints_limits=None,
                 base_batch_size=None,
                 moo_batch_size=None,
                 use_gpu=True,
                 ):
        model = get_one_output_model(model, output_layer_name)
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
                         x_test=x_test,
                         y_test=y_test,
                         base_model=models['base_model'],
                         moo_model=models['trainable_model'],
                         n_obj=n_obj,
                         eval_fs=eval_fs,
                         base_batch_size=base_batch_size,
                         moo_batch_size=moo_batch_size,
                         x_tol_for_hash=x_tol_for_hash,
                         use_gpu=use_gpu,
                         constraints_limits=constraints_limits)
