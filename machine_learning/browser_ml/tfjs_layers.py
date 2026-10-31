"""Keras teaching layers implemented on TensorFlow.js Layers in the browser.

They share the established Dense layer's weight methods and JS life cycle.
Only the documented arguments below are supported; unknown options fail loudly.
"""
from __future__ import annotations

from . import tfjs_bridge as bridge


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    result = tuple(value)
    if len(result) != 2 or any(not isinstance(x, int) or x <= 0 for x in result):
        raise ValueError("Expected a positive integer or a pair of positive integers")
    return result


def _options(layer, inferred_shape):
    out = {}
    shape = layer.input_shape or inferred_shape
    if shape is not None:
        out["inputShape"] = list(shape)
    if layer.name is not None:
        out["name"] = layer.name
    return out


class Flatten(bridge.Dense):
    def __init__(self, input_shape=None, name=None, data_format=None):
        if data_format not in (None, "channels_last"):
            raise NotImplementedError("Flatten supports channels_last")
        super().__init__(0, input_shape=input_shape, name=name)

    def _js_layer(self, inferred_input=None):
        if self._layer is None:
            self._layer = bridge._require_tf().layers.flatten(
                bridge._config(**_options(self, inferred_input)))
        return self._layer


class Dropout(bridge.Dense):
    def __init__(self, rate, input_shape=None, name=None, seed=None):
        if not 0 <= float(rate) < 1:
            raise ValueError("Dropout rate must be in [0,1)")
        super().__init__(0, input_shape=input_shape, name=name)
        self.rate = float(rate)
        self.seed = seed

    def _js_layer(self, inferred_input=None):
        if self._layer is None:
            opts = _options(self, inferred_input)
            opts["rate"] = self.rate
            if self.seed is not None:
                opts["seed"] = int(self.seed)
            self._layer = bridge._require_tf().layers.dropout(bridge._config(**opts))
        return self._layer


class Conv2D(bridge.Dense):
    def __init__(self, filters, kernel_size, strides=(1,1), padding="valid",
                 activation="linear", input_shape=None, name=None, use_bias=True,
                 data_format="channels_last", kernel_regularizer=None):
        if data_format != "channels_last":
            raise NotImplementedError("Conv2D supports channels_last")
        if padding not in ("valid", "same"):
            raise ValueError("Unsupported Conv2D padding")
        super().__init__(filters, activation=activation, input_shape=input_shape,
                         name=name, kernel_regularizer=kernel_regularizer)
        self.filters = int(filters)
        self.kernel_size = _pair(kernel_size)
        self.strides = _pair(strides)
        self.padding = padding
        self.use_bias = bool(use_bias)

    def _js_layer(self, inferred_input=None):
        if self._layer is None:
            tf = bridge._require_tf()
            opts = _options(self, inferred_input)
            opts.update(filters=self.filters, kernelSize=list(self.kernel_size),
                        strides=list(self.strides), padding=self.padding,
                        activation=self._activation_name, useBias=self.use_bias)
            if self.kernel_regularizer is not None:
                strength = getattr(self.kernel_regularizer, "l2", None)
                if strength is None:
                    raise TypeError("Only L2 kernel_regularizer is supported")
                opts["kernelRegularizer"] = tf.regularizers.l2(
                    bridge._config(l2=float(strength)))
            self._layer = tf.layers.conv2d(bridge._config(**opts))
        return self._layer


class MaxPooling2D(bridge.Dense):
    def __init__(self, pool_size=(2,2), strides=None, padding="valid",
                 input_shape=None, name=None, data_format="channels_last"):
        if data_format != "channels_last":
            raise NotImplementedError("MaxPooling2D supports channels_last")
        if padding not in ("valid", "same"):
            raise ValueError("Unsupported pooling padding")
        super().__init__(0, input_shape=input_shape, name=name)
        self.pool_size = _pair(pool_size)
        self.strides = _pair(strides) if strides is not None else self.pool_size
        self.padding = padding

    def _js_layer(self, inferred_input=None):
        if self._layer is None:
            opts = _options(self, inferred_input)
            opts.update(poolSize=list(self.pool_size),
                        strides=list(self.strides), padding=self.padding)
            self._layer = bridge._require_tf().layers.maxPooling2d(
                bridge._config(**opts))
        return self._layer


class BatchNormalization(bridge.Dense):
    def __init__(self, axis=-1, momentum=0.99, epsilon=0.001,
                 input_shape=None, name=None):
        if axis != -1:
            raise NotImplementedError("BatchNormalization supports axis=-1")
        super().__init__(0, input_shape=input_shape, name=name)
        self.axis = axis
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)

    def _js_layer(self, inferred_input=None):
        if self._layer is None:
            opts = _options(self, inferred_input)
            opts.update(axis=self.axis, momentum=self.momentum, epsilon=self.epsilon)
            self._layer = bridge._require_tf().layers.batchNormalization(
                bridge._config(**opts))
        return self._layer
