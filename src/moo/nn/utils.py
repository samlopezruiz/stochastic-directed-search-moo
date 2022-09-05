import numpy as np
import tensorflow as tf
from tensorflow.keras.models import clone_model


def params_conversion_weights(weights):
    shapes = [w.shape for w in weights]
    flatten_dim = [np.multiply(*s) if len(s) > 1 else s[0] for s in shapes]

    ind = np.concatenate([w.flatten() for w in weights]).reshape(1, -1)
    params = {
        'shapes': shapes,
        'flatten_dim': flatten_dim
    }
    return ind, params


def reconstruct_weights(ind, params):
    shapes, flatten_dim = params['shapes'], params['flatten_dim']
    reconstruct = []
    ind = ind.reshape(-1, )
    flatten_dim = np.cumsum(flatten_dim)
    flatten_dim = np.insert(flatten_dim, 0, 0)
    for i in range(len(shapes)):
        reconstruct.append(ind[flatten_dim[i]:flatten_dim[i + 1]].reshape(shapes[i]))

    return reconstruct


def get_last_layer_weights(model, layer_name='moo_layer'):
    relevant_layers = [l for l in model.layers if layer_name in l.name]
    if len(relevant_layers) > 1:
        raise Exception('More than one layer found')
    else:
        last_layer = relevant_layers[0]
        return last_layer.get_weights(), last_layer


def get_moo_layer(model, layer_name='moo_layer'):
    relevant_layers = []
    for i, l in enumerate(model.layers):
        if layer_name in l.name:
            relevant_layers.append((i, l))
    if len(relevant_layers) > 1:
        raise Exception('More than one layer found')
    else:
        i, last_layer = relevant_layers[0]
        # return {'weights': last_layer.get_weights(), 'ix': i, 'last_layer': last_layer}
        return last_layer.get_weights(), last_layer


def batch_array(arr, batch_size=None):
    if batch_size is None:
        return [arr]

    batches = []
    for i in range((arr.shape[0] // batch_size) + 1):
        batches.append(arr[i * batch_size:min(i * batch_size + batch_size, arr.shape[0]), Ellipsis])

    return batches


def batch_from_list_or_array(input_, batch_size=None):
    if isinstance(input_, list):
        batched_arrs = [batch_array(arr, batch_size) for arr in input_]
        batched = []
        for i in range(len(batched_arrs[0])):
            batched.append([batch[i] for batch in batched_arrs])
    else:
        batched = batch_array(input_, batch_size)
    return batched


def predict_from_batches(model,
                         batches,
                         to_numpy=True,
                         concat_output=True,
                         use_gpu=True):
    with tf.device('/device:GPU:0' if use_gpu else "/cpu:0"):
        outputs = []
        for batch in batches:
            pred = model(batch)
            if isinstance(pred, list):
                outputs.append([(out.numpy() if to_numpy else out) for out in pred])
            else:
                outputs.append(pred.numpy() if to_numpy else pred)

        if concat_output:
            if isinstance(outputs[0], list):
                concat_outputs = []
                for i in range(len(outputs[0])):
                    concat_outputs.append(np.concatenate([out[i] for out in outputs], axis=0) if to_numpy
                                          else tf.concat([out[i] for out in outputs], axis=0))
            else:
                concat_outputs = np.concatenate(outputs, axis=0) if to_numpy else tf.concat(outputs, axis=0)
            return concat_outputs
        else:
            return outputs


def get_one_output_model(model, output_layer_name):
    return tf.keras.Model(inputs=model.inputs,
                          outputs=model.get_layer(output_layer_name).output)


def split_model(model, intermediate_layers):
    base_model = tf.keras.Model(inputs=model.inputs,
                                outputs=[model.get_layer(l).output for l in intermediate_layers])

    trainable_model = tf.keras.Model(inputs=[model.get_layer(l).output for l in intermediate_layers],
                                     outputs=model.outputs, )

    base_model.compile()
    trainable_model.compile()
    return {'base_model': base_model,
            'trainable_model': trainable_model}
