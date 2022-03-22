import tensorflow as tf

# Use the GitHub version of TFCO
# !pip install git+https://github.com/google-research/tensorflow_constrained_optimization
import tensorflow_constrained_optimization as tfco

if __name__ == '__main__':

    class SampleProblem(tfco.ConstrainedMinimizationProblem):
        def __init__(self, loss_fn, weights):
            self._loss_fn = loss_fn
            self._weights = weights

        @property
        def num_constraints(self):
            return 4

        def objective(self):
            return loss_fn()

        def constraints(self):
            x, y = self._weights
            sum_weights = x + y
            lt_or_eq_one = sum_weights - 1
            gt_or_eq_one = 1 - sum_weights
            constraints = tf.stack([lt_or_eq_one, gt_or_eq_one, -x, -y])
            return constraints


    x = tf.Variable(0.0, dtype=tf.float32, name='x')
    y = tf.Variable(0.0, dtype=tf.float32, name='y')


    def loss_fn():
        return (x - 2) ** 2 + y


    problem = SampleProblem(loss_fn, [x, y])

    optimizer = tfco.LagrangianOptimizer(
        optimizer=tf.optimizers.Adagrad(learning_rate=0.1),
        num_constraints=problem.num_constraints
    )

    var_list = list(problem.trainable_variables) + optimizer.trainable_variables()

    for i in range(10000):
        optimizer.minimize(problem, var_list=var_list)
        if i % 1000 == 0:
            print(f'step = {i}')
            print(f'loss = {loss_fn()}')
            print(f'constraint = {(x + y).numpy()}')
            print(f'x = {x.numpy()}, y = {y.numpy()}')
