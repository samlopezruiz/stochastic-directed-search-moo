import tensorflow as tf

from src.sds.nn.problem import TFSplitProblem
from src.sds.nn.utils import predict_from_batches


class TsQuantileProblem(TFSplitProblem):

    def __init__(self,
                 y_train,
                 x_train,
                 model,
                 eval_fs,
                 quantile_ix=None,
                 x_valid=None,
                 y_valid=None,
                 x_test=None,
                 y_test=None,
                 x_tol_for_hash=None,
                 constraints_limits=None,
                 base_batch_size=None,
                 moo_batch_size=None,
                 moo_model_size='small',
                 use_gpu=True,
                 ):

        self.quantile_ix = quantile_ix

        if moo_model_size == 'medium':
            intermediate_layers = ['layer_normalization_36', 'time_distributed_144']
        elif moo_model_size == 'small':
            intermediate_layers = ['layer_normalization_40']
        else:
            raise NotImplementedError

        super().__init__(y_train=y_train,
                         x_train=x_train,
                         x_valid=x_valid,
                         y_valid=y_valid,
                         x_test=x_test,
                         y_test=y_test,
                         model=model,
                         output_layer_name='td_quantiles',
                         intermediate_layers=intermediate_layers,
                         n_obj=4 if self.quantile_ix is None else 2,
                         eval_fs=eval_fs,
                         base_batch_size=base_batch_size,
                         moo_batch_size=moo_batch_size,
                         x_tol_for_hash=x_tol_for_hash,
                         use_gpu=use_gpu,
                         constraints_limits=constraints_limits)

    def _get_losses(self, x_batches, y):
        y_pred = predict_from_batches(self.moo_model,
                                      x_batches,
                                      to_numpy=False,
                                      concat_output=True,
                                      use_gpu=self.use_gpu)
        loss = self.eval_fs[0](y, y_pred)
        if self.quantile_ix is None:
            losses = []
            [[losses.append(l.numpy()) for l in loss[i]] for i in [0, 2]]
            return losses
        else:
            return [l.numpy() for l in loss[self.quantile_ix]]

    def _batch_gradient(self, x, y):
        with tf.device('/device:GPU:0' if self.use_gpu else "/cpu:0"):
            with tf.GradientTape(persistent=True) as tape:
                # grads are calculated always with train data
                y_pred = self.moo_model(x)
                loss = self.eval_fs[0](y, y_pred)

            if self.quantile_ix is None:
                grads = []
                [[grads.append(tape.gradient(l, self.moo_model.trainable_variables)) for l in loss[i]] for i in [0, 2]]
                return grads
            else:
                grads = [tape.gradient(loss[self.quantile_ix][i], self.moo_model.trainable_variables) for i in [0, 1]]
        return grads
