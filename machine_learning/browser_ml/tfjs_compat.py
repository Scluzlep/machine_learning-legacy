"""Course-oriented TensorFlow / tf.keras API compatibility in Pyodide.

The custom Pyodide kernel installs this module automatically on the first
`import tensorflow`; course notebooks keep their original Python syntax.
Unsupported operations fail explicitly rather than returning wrong math.
"""
from __future__ import annotations

import sys
import types
import warnings
import numpy as np
from js import globalThis
from . import tfjs_bridge as bridge
from . import tfjs_tensor as eager
from . import tfjs_layers as extra_layers


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value

    def __array__(self, dtype=None, copy=None):
        if copy is None:
            return np.asarray(self.value, dtype=dtype)
        return np.array(self.value, dtype=dtype, copy=copy)

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def item(self):
        return self.value.item()

    def __float__(self):
        if self.value.size != 1:
            raise TypeError("Only one-element tensors can be converted to float")
        return float(self.value.item())

    def __int__(self):
        if self.value.size != 1:
            raise TypeError("Only one-element tensors can be converted to int")
        return int(self.value.item())

    def __repr__(self):
        return repr(self.value)

    def __getitem__(self, item):
        return self.value[item]


def _sigmoid(x):
    v = np.asarray(x, dtype=np.float64)
    return _Tensor(1 / (1 + np.exp(-np.clip(v, -100, 100))))


def _softmax(x, axis=-1):
    v = np.asarray(x, dtype=np.float64)
    z = np.exp(v - np.max(v, axis=axis, keepdims=True))
    return _Tensor(z / np.sum(z, axis=axis, keepdims=True))


def _relu(x):
    return _Tensor(np.maximum(np.asarray(x), 0))


def _linear(x):
    return _Tensor(np.asarray(x))


_ACTIVATIONS = {
    "relu": _relu, "linear": _linear, "sigmoid": _sigmoid, "softmax": _softmax
}


class _TensorShape:
    def __init__(self, dimensions):
        self._dimensions = list(dimensions)

    def as_list(self):
        return list(self._dimensions)


class _SymbolicTensor:
    def __init__(self, dimensions):
        self.shape = _TensorShape(dimensions)


class Dense(bridge.Dense):
    """Keras metadata uses Python functions; JS layers use activation names."""

    def __init__(self, units, activation="linear", **kwargs):
        if callable(activation):
            choices = [k for k, v in _ACTIVATIONS.items() if v is activation]
            if not choices:
                raise TypeError("Unsupported activation callable")
            activation = choices[0]
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation {activation!r}")
        super().__init__(units, activation=activation, **kwargs)
        self.activation = _ACTIVATIONS[activation]

    @property
    def output(self):
        if not hasattr(self, "_output_shape"):
            raise ValueError("Build model before inspecting layer.output")
        return _SymbolicTensor(self._output_shape)


class l2:
    """Keras-compatible L2 regularization configuration."""
    def __init__(self, l2=0.01):
        self.l2 = float(l2)
        if self.l2 < 0:
            raise ValueError("L2 must be nonnegative")


_floatx = "float32"


def set_floatx(value):
    global _floatx
    if value not in ("float32", "float64"):
        raise ValueError("Only float32/float64 accepted")
    if value == "float64":
        warnings.warn(
            "Keras requested float64, but TensorFlow.js Dense kernels use float32",
            RuntimeWarning, stacklevel=2,
        )
    _floatx = value


def floatx():
    return _floatx


class Input:
    def __init__(self, shape=None, **kwargs):
        self.shape = tuple(shape) if shape is not None else None


class Normalization:
    def __init__(self, axis=-1, **kwargs):
        if axis != -1:
            raise NotImplementedError("Normalization currently supports axis=-1")
        self.mean = None
        self.std = None

    def adapt(self, data):
        data = np.asarray(data, dtype=np.float32)
        self.mean = data.mean(axis=0)
        self.std = np.sqrt(data.var(axis=0) + 1e-7)

    def __call__(self, data):
        if self.mean is None:
            raise ValueError("Call adapt() before using Normalization")
        return (np.asarray(data, dtype=np.float32) - self.mean) / self.std


class LeakyReLU:
    """Import-compatible activation; model-layer support is not yet enabled."""
    def __init__(self, negative_slope=0.3, alpha=None, **kwargs):
        self.alpha = float(negative_slope if alpha is None else alpha)

    def __call__(self, x):
        x = np.asarray(x)
        return np.maximum(x, self.alpha * x)


class Adam:
    def __init__(self, learning_rate=0.001, lr=None, **kwargs):
        self.learning_rate = _Tensor(float(lr if lr is not None else learning_rate))


class _Loss:
    tfjs_name = None
    def __init__(self, from_logits=False, **kwargs):
        if kwargs:
            raise TypeError("Unsupported loss options: " + ", ".join(kwargs))
        self.from_logits = bool(from_logits)

    def get_config(self):
        return {"from_logits": self.from_logits}


class BinaryCrossentropy(_Loss):
    tfjs_name = "binaryCrossentropy"


class MeanSquaredError(_Loss):
    tfjs_name = "meanSquaredError"


class SparseCategoricalCrossentropy(_Loss):
    tfjs_name = "sparseCategoricalCrossentropy"


class CategoricalCrossentropy(_Loss):
    tfjs_name = "categoricalCrossentropy"


class History:
    """Keras History object, exposing history.history['loss']."""
    def __init__(self, data):
        self.history = data

    def __repr__(self):
        return f"History({self.history!r})"


class Sequential:
    def __init__(self, layers=None, name="model"):
        layer_defs = []
        input_shape = None
        for layer in layers or []:
            if isinstance(layer, Input):
                input_shape = layer.shape
            elif isinstance(layer, bridge.Dense):
                if not layer_defs and input_shape is not None and layer.input_shape is None:
                    layer.input_shape = input_shape
                layer_defs.append(layer)
            else:
                raise TypeError(
                    f"Unsupported layer {type(layer).__name__}; "
                    "this adapter currently trains Dense networks"
                )
        self._model = bridge.Sequential(layer_defs, name=name)
        self._sparse_target_classes = None
        self.loss = None
        self.optimizer = None

    @property
    def layers(self):
        return self._model.layers

    def add(self, layer):
        if not isinstance(layer, bridge.Dense):
            raise TypeError("Only Dense layers can be added")
        self._model.add(layer)

    def summary(self):
        self._model.summary()

    def build(self, input_shape):
        if input_shape is None or len(input_shape) < 2 or input_shape[-1] is None:
            raise ValueError("Expected input shape (None, n_features)")
        self._model._build(int(input_shape[-1]))

    @property
    def input(self):
        shape = self._model._built_input_shape
        if shape is None:
            raise ValueError("Build model before inspecting model.input")
        return _SymbolicTensor((None, *shape))

    def get_layer(self, name=None, index=None):
        layers = self.layers
        if index is not None:
            return layers[index]
        for layer in layers:
            if layer.name == name:
                return layer
        raise ValueError(f"No layer named {name!r}")

    def get_weights(self):
        return self._model.get_weights()

    def compile(self, loss=None, optimizer=None, metrics=None, **kwargs):
        if kwargs:
            raise TypeError("Unsupported compile arguments: " + ", ".join(kwargs))
        if loss is None:
            loss = MeanSquaredError()
        if isinstance(loss, str):
            # Keras 2 snake_case loss names differ from TF.js camelCase names.
            names = {
                "mse": MeanSquaredError, "meansquarederror": MeanSquaredError,
                "binarycrossentropy": BinaryCrossentropy,
                "categoricalcrossentropy": CategoricalCrossentropy,
                "sparsecategoricalcrossentropy": SparseCategoricalCrossentropy,
            }
            factory = names.get(loss.replace("_", "").lower())
            if factory is not None:
                loss = factory()
        if optimizer is None or optimizer == "adam":
            optimizer = Adam()
        if not isinstance(optimizer, Adam):
            raise TypeError("Only Adam is currently supported")
        self._sparse_target_classes = None
        self.loss = loss
        self.optimizer = optimizer
        if isinstance(loss, _Loss):
            if isinstance(loss, SparseCategoricalCrossentropy):
                if not self.layers:
                    raise ValueError("Sparse categorical loss requires an output layer")
                self._sparse_target_classes = self.layers[-1].units
                fn_name = ("sparseCategoricalCrossentropyFromLogits"
                           if loss.from_logits else
                           "sparseCategoricalCrossentropyProbabilities")
                loss_value = getattr(globalThis.mlCourseLosses, fn_name)
            elif loss.from_logits:
                fn_name = {
                    BinaryCrossentropy: "binaryCrossentropyFromLogits",
                }.get(type(loss))
                if not fn_name:
                    raise NotImplementedError(
                        f"{type(loss).__name__}(from_logits=True) is not supported yet"
                    )
                loss_value = getattr(globalThis.mlCourseLosses, fn_name)
            else:
                loss_value = loss.tfjs_name
        elif isinstance(loss, str):
            loss_value = loss
        else:
            raise TypeError("Unsupported loss: " + repr(loss))
        self._model.compile(loss=loss_value,
                            learning_rate=float(optimizer.learning_rate), metrics=metrics)

    async def fit(self, X, y, epochs=10, batch_size=32,
                  verbose=0, shuffle=True, **kwargs):
        if kwargs:
            raise TypeError("Unsupported fit arguments: " + ", ".join(kwargs))
        # TF.js Layers validates y shape before invoking a custom loss.
        # Convert Keras integer labels to the expected one-hot tensor here;
        # the original Notebook and its NumPy integer labels stay unchanged.
        if self._sparse_target_classes is not None:
            values = np.asarray(y).reshape(-1)
            if len(values) != len(X):
                raise ValueError("Classification labels must match X rows")
            n_classes = self._sparse_target_classes
            if np.any(values < 0) or np.any(values >= n_classes):
                raise ValueError("Labels must be in 0..number_of_classes-1")
            if not np.all(values == values.astype(np.int32)):
                raise ValueError("Sparse categorical labels must be integers")
            y = np.eye(n_classes, dtype=np.float32)[values.astype(np.int32)]
        data = await self._model.fit(X, y, epochs=epochs,
                                     batch_size=batch_size, verbose=verbose,
                                     shuffle=shuffle)
        return History(data)

    def predict(self, X, **kwargs):
        return self._model.predict(X, **kwargs)

    async def predict_async(self, X):
        return await self._model.predict_async(X)

    async def get_weights_async(self):
        return await self._model.get_weights_async()

    def __call__(self, X, **kwargs):
        return _Tensor(self.predict(X))

    def dispose(self):
        self._model.dispose()


def _module(name, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    sys.modules[name] = module
    return module


async def install(backend="auto"):
    """Expose the supported TensorFlow 2/Keras course API as Python modules."""
    await bridge.load_tfjs(backend=backend)
    if getattr(globalThis, "mlCourseLosses", None) is None:
        await bridge._load_script(bridge._asset_url("course-losses.js"))
    layers = _module("tensorflow.keras.layers",
                     Dense=Dense, Input=Input, Normalization=Normalization,
                     LeakyReLU=LeakyReLU,
                     Conv2D=extra_layers.Conv2D, MaxPooling2D=extra_layers.MaxPooling2D,
                     Flatten=extra_layers.Flatten, Dropout=extra_layers.Dropout,
                     BatchNormalization=extra_layers.BatchNormalization)
    activations = _module(
        "tensorflow.keras.activations",
        sigmoid=_sigmoid, relu=_relu,
        linear=_linear,
        softmax=_softmax
    )
    losses = _module(
        "tensorflow.keras.losses",
        BinaryCrossentropy=BinaryCrossentropy,
        MeanSquaredError=MeanSquaredError,
        SparseCategoricalCrossentropy=SparseCategoricalCrossentropy,
        CategoricalCrossentropy=CategoricalCrossentropy,
    )
    optimizers = _module("tensorflow.keras.optimizers", Adam=Adam)
    regularizers = _module("tensorflow.keras.regularizers", l2=l2)
    keras_backend = _module("tensorflow.keras.backend", set_floatx=set_floatx, floatx=floatx)
    models = _module("tensorflow.keras.models", Sequential=Sequential)
    keras = _module("tensorflow.keras", Sequential=Sequential, Input=Input,
                    layers=layers, activations=activations, losses=losses,
                    optimizers=optimizers, regularizers=regularizers, backend=keras_backend,
                     models=models, __version__="2.6.0+tfjs-compat")
    # TensorFlow.js-backed eager operations. NumPy inputs preserve the existing
    # teaching-course semantics for tf.nn.softmax and Keras activations, while
    # GPU Tensor inputs stay on the active backend (and remain differentiable).
    def nn_softmax(value, axis=-1):
        return eager.softmax(value, axis) if isinstance(value, eager.Tensor) else _softmax(value, axis)

    def nn_sigmoid(value):
        return eager.sigmoid(value) if isinstance(value, eager.Tensor) else _sigmoid(value)

    def nn_relu(value):
        return eager.relu(value) if isinstance(value, eager.Tensor) else _relu(value)

    math = _module("tensorflow.math", exp=eager.exp, log=eager.log,
                   sqrt=eager.sqrt, square=eager.square, sigmoid=nn_sigmoid,
                   tanh=eager.tanh, abs=eager.abs, maximum=eager.maximum,
                   minimum=eager.minimum)
    linalg = _module("tensorflow.linalg", matmul=eager.matmul)
    nn = _module("tensorflow.nn", softmax=nn_softmax, sigmoid=nn_sigmoid,
                 relu=nn_relu)
    random = _module("tensorflow.random", set_seed=np.random.seed,
                     normal=eager.random_normal, uniform=eager.random_uniform)
    tf = _module(
        "tensorflow", keras=keras, Tensor=eager.Tensor, Variable=eager.Variable,
        float32=eager.float32, int32=eager.int32, bool=eager.bool,
        constant=eager.constant, convert_to_tensor=eager.convert_to_tensor,
        is_tensor=eager.is_tensor, cast=eager.cast, matmul=eager.matmul,
        add=eager.add, subtract=eager.subtract, multiply=eager.multiply,
        divide=eager.divide, square=eager.square, sqrt=eager.sqrt,
        exp=eager.exp, reshape=eager.reshape, transpose=eager.transpose,
        expand_dims=eager.expand_dims, squeeze=eager.squeeze,
        reduce_sum=eager.reduce_sum, reduce_mean=eager.reduce_mean,
        reduce_max=eager.reduce_max, reduce_min=eager.reduce_min,
        ones=eager.ones, zeros=eager.zeros, ones_like=eager.ones_like,
        zeros_like=eager.zeros_like, eye=eager.eye, stack=eager.stack,
        concat=eager.concat, one_hot=eager.one_hot,
        clip_by_value=eager.clip_by_value, math=math, linalg=linalg,
        nn=nn, random=random,
        __version__="2.6.0+tfjs-compat",
        autograph=types.SimpleNamespace(set_verbosity=lambda *_: None),
    )
    return tf
