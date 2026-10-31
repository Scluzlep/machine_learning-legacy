"""Python bridge to the TensorFlow.js model trainer bundled with this website.

Usage (Pyodide kernel):
    from browser_ml.tfjs_bridge import load_tfjs, Dense, Sequential
    await load_tfjs()
    model = Sequential([Dense(8, "relu", input_shape=(2,)), Dense(1, "sigmoid")])
    model.compile(loss="binaryCrossentropy", learning_rate=0.01)
    history = await model.fit(X, y, epochs=40)
    yhat = model.predict(X)
"""
from __future__ import annotations

from urllib.parse import urlsplit

import numpy as np

try:
    from js import globalThis
    from pyodide.ffi import to_js
except ImportError:
    globalThis = None
    to_js = None

_tf = None
_webgpu_loaded = False
_backend_attempts = []


def _config(**kwargs):
    return to_js(kwargs, dict_converter=globalThis.Object.fromEntries)


def _asset_url(asset_name="tf.min.js"):
    loc = urlsplit(str(globalThis.location.href))
    path = loc.path
    for marker in ("/lab/", "/notebooks/", "/tree/", "/api/", "/extensions/", "/build/"):
        if marker in path:
            path = path.split(marker, 1)[0]
            break
    else:
        path = path.rstrip("/").rsplit("/", 1)[0]
    return f"{loc.scheme}://{loc.netloc}{path}/vendor/{asset_name}"



async def _load_script(url):
    """Load a locally hosted UMD bundle into a JupyterLite Pyodide worker."""
    try:
        globalThis.importScripts(url)
    except Exception:
        response = await globalThis.fetch(url)
        if not response.ok:
            raise RuntimeError(f"JavaScript asset failed: {url} (HTTP {response.status})")
        globalThis.eval(await response.text())


def _worker_has_webgpu():
    try:
        return getattr(getattr(globalThis, "navigator", None), "gpu", None) is not None
    except Exception:
        return False


async def _activate_backend(name):
    """Activate the backend and verify that it can run a GPU/CPU operation."""
    tf = _require_tf()
    result = await tf.setBackend(name)
    if not result:
        raise RuntimeError(f"Backend {name} could not be activated")
    await tf.ready()
    x = tf.tensor2d(to_js([[1.0, 2.0], [3.0, 4.0]]))
    try:
        y = x.matMul(x)
        try:
            await y.data()
        finally:
            y.dispose()
    finally:
        x.dispose()
    if str(tf.getBackend()) != name:
        raise RuntimeError(f"Backend {name} not selected")


async def load_tfjs(url=None, backend="auto", webgpu_url=None):
    """Initialize TF.js and choose WebGPU -> WebGL -> CPU.

    Choose backend="webgpu" to require WebGPU (raises if unavailable);
    use backend="cpu" for deterministic CPU comparison.
    """
    global _tf, _webgpu_loaded, _backend_attempts
    if backend not in ("auto", "webgpu", "webgl", "cpu"):
        raise ValueError("backend must be 'auto', 'webgpu', 'webgl' or 'cpu'")
    if globalThis is None:
        raise RuntimeError("TensorFlow.js requires the browser Pyodide kernel")
    if _tf is None:
        try:
            _tf = globalThis.tf
        except AttributeError:
            target = url or _asset_url()
            await _load_script(target)
            try:
                _tf = globalThis.tf
            except AttributeError as exc:
                raise RuntimeError(f"TensorFlow.js did not initialize: {target}") from exc
    _backend_attempts = []
    candidates = ("webgpu", "webgl", "cpu") if backend == "auto" else (backend,)
    for candidate in candidates:
        try:
            if candidate == "webgpu":
                if not _worker_has_webgpu():
                    raise RuntimeError("navigator.gpu unavailable in the kernel Worker")
                if not _webgpu_loaded:
                    await _load_script(webgpu_url or (url.rsplit("/", 1)[0] + "/tf-backend-webgpu.min.js" if url else _asset_url("tf-backend-webgpu.min.js")))
                    _webgpu_loaded = True
            await _activate_backend(candidate)
            return _tf
        except Exception as exc:
            _backend_attempts.append({"backend": candidate, "error": str(exc)})
            if backend != "auto":
                raise RuntimeError(f"Unable to use {candidate}: {exc}") from exc
    raise RuntimeError("No usable TensorFlow.js backend: " + str(_backend_attempts))


def backend_info():
    """Show active backend, WebGPU availability and fallback reasons."""
    return {
        "backend": backend(),
        "worker_webgpu_available": _worker_has_webgpu(),
        "fallbacks": list(_backend_attempts),
    }


def _require_tf():
    if _tf is None:
        raise RuntimeError("Call await load_tfjs() first")
    return _tf


def _tensor(value):
    data = np.asarray(value, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape((-1, 1))
    if data.ndim not in (2, 3, 4):
        raise ValueError("Layer inputs must have rank 2, 3, or 4")
    return _require_tf().tensor(to_js(data.ravel()), to_js(list(data.shape)), "float32")


class Dense:
    """Keras-style dense layer with optional delayed input-shape inference."""

    def __init__(self, units, activation="linear", input_shape=None,
                 name=None, input_dim=None, kernel_regularizer=None, **kwargs):
        if kwargs:
            raise TypeError("Unsupported Dense arguments: " + ", ".join(kwargs))
        self.units = int(units)
        self.activation = activation
        self._activation_name = activation
        self.kernel_regularizer = kernel_regularizer
        self.input_shape = tuple(input_shape) if input_shape is not None else (
            (int(input_dim),) if input_dim is not None else None
        )
        self.name = name
        self._layer = None

    def _js_layer(self, inferred_input=None):
        if self._layer is None:
            tf = _require_tf()
            opts = {"units": self.units, "activation": self._activation_name}
            if self.kernel_regularizer is not None:
                strength = getattr(self.kernel_regularizer, "l2", None)
                if strength is None:
                    raise TypeError("Only L2 kernel_regularizer is implemented")
                opts["kernelRegularizer"] = tf.regularizers.l2(_config(l2=float(strength)))
            shape = self.input_shape or inferred_input
            if shape is not None:
                opts["inputShape"] = list(shape)
            if self.name is not None:
                opts["name"] = self.name
            self._layer = tf.layers.dense(_config(**opts))
        return self._layer

    def __call__(self, value):
        x = _tensor(value)
        try:
            result = self._js_layer().apply(x)
            try:
                return np.asarray(result.arraySync().to_py(), dtype=np.float32)
            finally:
                result.dispose()
        finally:
            x.dispose()

    def get_weights(self):
        if self._layer is None:
            return []
        return [np.asarray(t.arraySync().to_py(), dtype=np.float32)
                for t in self._layer.getWeights()]

    def set_weights(self, values):
        if self._layer is None:
            raise ValueError("The layer has not been built; call it or build the model first")
        tf = _require_tf()
        tensors = [tf.tensor(to_js(np.asarray(v, dtype=np.float32).tolist()))
                   for v in values]
        try:
            self._layer.setWeights(globalThis.Array.of(*tensors))
        finally:
            for tensor in tensors:
                tensor.dispose()

    @property
    def weights(self):
        return self.get_weights()


class Sequential:
    """TF.js Layers model built lazily when Keras omits input_shape."""

    def __init__(self, layers, name="model"):
        self._tf = _require_tf()
        self._name = name
        self._layers = list(layers)
        self._model = None
        self._compile_options = None
        self._built_n_features = None
        self._built_input_shape = None
        if not all(isinstance(layer, Dense) for layer in self._layers):
            raise TypeError("Supported layers: Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization")
        if self._layers and self._layers[0].input_shape is not None:
            self._build(self._layers[0].input_shape)

    def add(self, layer):
        if self._model is not None:
            raise RuntimeError("Adding layers after building the model is not supported")
        if not isinstance(layer, Dense):
            raise TypeError("Expected a Dense layer")
        self._layers.append(layer)

    @property
    def layers(self):
        return self._layers

    def _build(self, input_shape):
        shape = (int(input_shape),) if isinstance(input_shape, int) else tuple(input_shape)
        if not shape or any(int(dim) <= 0 for dim in shape):
            raise ValueError("Model input dimensions must be positive integers")
        if self._model is not None:
            if shape != self._built_input_shape:
                raise ValueError(
                    f"Input shape changed from {self._built_input_shape} to {shape}"
                )
            return
        if not self._layers:
            raise ValueError("Sequential model has no layers")
        self._model = self._tf.sequential(_config(name=self._name))
        previous_shape = (None, *shape)
        for i, layer in enumerate(self._layers):
            js_layer = layer._js_layer(shape if i == 0 else None)
            self._model.add(js_layer)
            layer._input_shape = previous_shape
            # TF.js layers expose symbolic output shapes through the model's
            # output tensor, not through the Python-Keras inputShape property.
            # Dense shape is statically known from its units; never infer it
            # from the Sequential model output, which may refer to another
            # symbolic layer when expanding a network.
            if type(layer).__name__ == "Dense":
                output_shape = (None, layer.units)
            else:
                output_shape = tuple(self._model.outputs[0].shape.to_py())
            layer._output_shape = output_shape
            previous_shape = output_shape
        self._built_input_shape = shape
        self._built_n_features = shape[0] if len(shape) == 1 else shape
        if self._compile_options is not None:
            self._model.compile(self._compile_options)

    def compile(self, *, optimizer="adam", learning_rate=0.001,
                loss="meanSquaredError", metrics=None):
        if optimizer != "adam":
            raise ValueError("Only the Adam optimizer is supported")
        options = {
            "optimizer": self._tf.train.adam(float(learning_rate)),
            "loss": loss,
        }
        if metrics is not None:
            options["metrics"] = metrics
        self._compile_options = _config(**options)
        if self._model is not None:
            self._model.compile(self._compile_options)

    async def fit(self, X, y, *, epochs=10, batch_size=32, verbose=0,
                  shuffle=True):
        data = np.asarray(X, dtype=np.float32)
        if data.ndim not in (2, 3, 4):
            raise ValueError("model.fit expects rank 2 to 4 input data")
        self._build(data.shape[1:])
        if self._compile_options is None:
            raise ValueError("Compile the model before calling fit")
        xjs = _tensor(data)
        yjs = _tensor(y)
        try:
            result = await self._model.fit(
                xjs, yjs,
                _config(epochs=int(epochs), batchSize=int(batch_size),
                        verbose=int(verbose), shuffle=bool(shuffle))
            )
            return result.history.to_py()
        finally:
            xjs.dispose()
            yjs.dispose()

    def predict(self, X, **kwargs):
        data = np.asarray(X, dtype=np.float32)
        if data.ndim not in (2, 3, 4):
            raise ValueError("model.predict expects rank 2 to 4 input data")
        self._build(data.shape[1:])
        xjs = _tensor(data)
        try:
            pred = self._model.predict(xjs)
            try:
                return np.asarray(pred.arraySync().to_py(), dtype=np.float32)
            finally:
                pred.dispose()
        finally:
            xjs.dispose()

    async def predict_async(self, X):
        data = np.asarray(X, dtype=np.float32)
        if data.ndim not in (2, 3, 4):
            raise ValueError("model.predict expects rank 2 to 4 input data")
        self._build(data.shape[1:])
        xjs = _tensor(data)
        try:
            pred = self._model.predict(xjs)
            try:
                return np.asarray((await pred.array()).to_py(), dtype=np.float32)
            finally:
                pred.dispose()
        finally:
            xjs.dispose()

    async def get_weights_async(self):
        if self._model is None:
            raise ValueError("Model has not been built yet")
        return [np.asarray((await tensor.array()).to_py(), dtype=np.float32)
                for tensor in self._model.getWeights()]

    def get_weights(self):
        if self._model is None:
            raise ValueError("Model has not been built yet")
        return [np.asarray(t.arraySync().to_py(), dtype=np.float32)
                for t in self._model.getWeights()]

    def summary(self):
        if self._model is None:
            print(f"Model: {self._name} (unbuilt, input shape inferred on first data)")
            for layer in self._layers:
                print(f"  Dense({layer.units}, activation={layer.activation!r})")
        else:
            self._model.summary()

    def dispose(self):
        if self._model is not None:
            self._model.dispose()

def backend():
    return str(_require_tf().getBackend())
