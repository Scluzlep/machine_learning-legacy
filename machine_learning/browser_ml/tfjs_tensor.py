"""TensorFlow.js-backed eager tensors for undergraduate ML/DL notebooks.

All mathematical operations run on the active TF.js backend. Python/NumPy
conversion is explicit and requires a GPU-to-CPU readback. The interface is a
carefully scoped subset of TensorFlow 2 eager mode, NOT full TensorFlow.
"""
from __future__ import annotations

import numpy as np
import builtins
from js import globalThis
from pyodide.ffi import to_js, create_proxy
from . import tfjs_bridge as bridge

float32 = "float32"
int32 = "int32"
bool = "bool"


def _tf():
    return bridge._require_tf()


def _dtype(value, dtype=None):
    if dtype is not None:
        name = np.dtype(dtype).name
        if name not in (float32, int32, bool):
            raise NotImplementedError(
                f"TensorFlow.js tensors support float32, int32, bool; got {name}"
            )
        return name
    dt = np.asarray(value).dtype
    if np.issubdtype(dt, np.bool_):
        return bool
    if np.issubdtype(dt, np.integer):
        return int32
    if np.issubdtype(dt, np.floating):
        return float32
    raise TypeError(f"Unsupported Tensor dtype: {dt}")


def _convert(value, dtype=None):
    if isinstance(value, Tensor):
        if dtype is not None and dtype != value.dtype:
            return cast(value, dtype)
        return value
    data_type = _dtype(value, dtype)
    data = np.asarray(value, dtype=data_type)
    # Explicit shape is essential for scalar and zero-sized tensors.
    js = _tf().tensor(to_js(data.ravel()), to_js(list(data.shape)), data_type)
    return Tensor(js)


def constant(value, dtype=None, shape=None, name=None):
    data = np.asarray(value)
    if shape is not None:
        size = int(np.prod(shape))
        if data.size == 1 and size != 1:
            data = np.full(shape, data.item())
        else:
            data = data.reshape(shape)
    return _convert(data, dtype=dtype)


def convert_to_tensor(value, dtype=None, name=None):
    return _convert(value, dtype)


def is_tensor(value):
    return isinstance(value, Tensor)


class Tensor:
    __array_priority__ = 1000

    def __init__(self, js_tensor, *, owned=True):
        self._js = js_tensor
        self._owned = owned
        self._disposed = False

    @property
    def shape(self):
        return tuple(self._js.shape.to_py())

    @property
    def dtype(self):
        return str(self._js.dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(self._js.size)

    def numpy(self):
        result = self._js.arraySync()
        return np.asarray(result.to_py() if hasattr(result, "to_py") else result,
                          dtype=self.dtype).reshape(self.shape)

    async def numpy_async(self):
        """Non-blocking WebGPU readback; preferred for large result tensors."""
        result = await self._js.array()
        return np.asarray(result.to_py() if hasattr(result, "to_py") else result,
                          dtype=self.dtype).reshape(self.shape)

    def __array__(self, dtype=None, copy=None):
        data = self.numpy()
        if copy is None:
            return np.asarray(data, dtype=dtype)
        return np.array(data, dtype=dtype, copy=copy)

    def item(self):
        return self.numpy().item()

    def __float__(self):
        if self.size != 1:
            raise TypeError("Only one-element Tensor can convert to float")
        return float(self.item())

    def __int__(self):
        if self.size != 1:
            raise TypeError("Only one-element Tensor can convert to int")
        return int(self.item())

    def __bool__(self):
        raise TypeError("Tensor truth value is ambiguous; use .numpy().item()")

    def __repr__(self):
        if self._disposed:
            return "<Tensor disposed>"
        return f"<tf.Tensor shape={self.shape} dtype={self.dtype}>"

    def dispose(self):
        if self._owned and not self._disposed:
            self._js.dispose()
        self._disposed = True

    def __del__(self):
        try:
            self.dispose()
        except Exception:
            pass

    def _binary(self, other, method):
        rhs = convert_to_tensor(other)
        return Tensor(getattr(_tf(), method)(self._js, rhs._js))

    def _rbinary(self, other, method):
        lhs = convert_to_tensor(other)
        return Tensor(getattr(_tf(), method)(lhs._js, self._js))

    def __add__(self, other): return self._binary(other, "add")
    def __radd__(self, other): return self._rbinary(other, "add")
    def __sub__(self, other): return self._binary(other, "sub")
    def __rsub__(self, other): return self._rbinary(other, "sub")
    def __mul__(self, other): return self._binary(other, "mul")
    def __rmul__(self, other): return self._rbinary(other, "mul")
    def __truediv__(self, other): return self._binary(other, "div")
    def __rtruediv__(self, other): return self._rbinary(other, "div")
    def __pow__(self, other): return self._binary(other, "pow")
    def __rpow__(self, other): return self._rbinary(other, "pow")
    def __matmul__(self, other): return matmul(self, other)
    def __rmatmul__(self, other): return matmul(other, self)
    def __neg__(self): return Tensor(_tf().neg(self._js))
    def __abs__(self): return Tensor(_tf().abs(self._js))
    def __lt__(self, other): return self._binary(other, "less")
    def __le__(self, other): return self._binary(other, "lessEqual")
    def __gt__(self, other): return self._binary(other, "greater")
    def __ge__(self, other): return self._binary(other, "greaterEqual")

    def __getitem__(self, key):
        # Indexing a tensor performs a CPU readback. Do not use this inside grad.
        return self.numpy()[key]

    def reshape(self, shape):
        return reshape(self, shape)

    def transpose(self, perm=None):
        return transpose(self, perm)


class Variable(Tensor):
    def __init__(self, initial_value, trainable=True, dtype=None, name=None):
        source = convert_to_tensor(initial_value, dtype=dtype)
        js = _tf().variable(source._js, builtins.bool(trainable), name)
        super().__init__(js)
        self.trainable = builtins.bool(trainable)
        self.name = str(js.name)

    def assign(self, value):
        replacement = convert_to_tensor(value, dtype=self.dtype)
        if replacement.shape != self.shape:
            raise ValueError("Variable.assign shape mismatch")
        self._js.assign(replacement._js)
        return self

    def assign_add(self, delta):
        value = self + delta
        try:
            return self.assign(value)
        finally:
            value.dispose()

    def assign_sub(self, delta):
        value = self - delta
        try:
            return self.assign(value)
        finally:
            value.dispose()


def _unary(name, value):
    data = convert_to_tensor(value)
    return Tensor(getattr(_tf(), name)(data._js))


def _binary(name, a, b):
    lhs, rhs = convert_to_tensor(a), convert_to_tensor(b)
    return Tensor(getattr(_tf(), name)(lhs._js, rhs._js))


def add(a, b): return _binary("add", a, b)
def subtract(a, b): return _binary("sub", a, b)
def multiply(a, b): return _binary("mul", a, b)
def divide(a, b): return _binary("div", a, b)
def maximum(a, b): return _binary("maximum", a, b)
def minimum(a, b): return _binary("minimum", a, b)
def equal(a, b): return _binary("equal", a, b)
def square(x): return _unary("square", x)
def sqrt(x): return _unary("sqrt", x)
def exp(x): return _unary("exp", x)
def log(x): return _unary("log", x)
def abs(x): return _unary("abs", x)
def sigmoid(x): return _unary("sigmoid", x)
def relu(x): return _unary("relu", x)
def tanh(x): return _unary("tanh", x)


def matmul(a, b, transpose_a=False, transpose_b=False):
    a, b = convert_to_tensor(a), convert_to_tensor(b)
    return Tensor(_tf().matMul(a._js, b._js, builtins.bool(transpose_a), builtins.bool(transpose_b)))


def reshape(tensor, shape, name=None):
    x = convert_to_tensor(tensor)
    return Tensor(_tf().reshape(x._js, to_js(list(shape))))


def transpose(a, perm=None, conjugate=False, name=None):
    if conjugate:
        raise NotImplementedError("Complex tensor conjugation not supported")
    x = convert_to_tensor(a)
    if perm is None:
        return Tensor(_tf().transpose(x._js))
    return Tensor(_tf().transpose(x._js, to_js(list(perm))))


def expand_dims(input, axis, name=None):
    x = convert_to_tensor(input)
    return Tensor(_tf().expandDims(x._js, int(axis)))


def squeeze(input, axis=None, name=None):
    x = convert_to_tensor(input)
    if axis is None:
        return Tensor(_tf().squeeze(x._js))
    axes = [axis] if isinstance(axis, int) else list(axis)
    return Tensor(_tf().squeeze(x._js, to_js(axes)))


def cast(x, dtype, name=None):
    x = convert_to_tensor(x)
    name = _dtype(np.array(0), dtype)
    if x.dtype == name:
        return x
    return Tensor(_tf().cast(x._js, name))


def _axes(axis):
    if axis is None or isinstance(axis, int):
        return axis
    return to_js(list(axis))


def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
    x = convert_to_tensor(input_tensor)
    return Tensor(_tf().sum(x._js, _axes(axis), builtins.bool(keepdims)))


def reduce_mean(input_tensor, axis=None, keepdims=False, name=None):
    x = convert_to_tensor(input_tensor)
    return Tensor(_tf().mean(x._js, _axes(axis), builtins.bool(keepdims)))


def reduce_max(input_tensor, axis=None, keepdims=False, name=None):
    x = convert_to_tensor(input_tensor)
    return Tensor(_tf().max(x._js, _axes(axis), builtins.bool(keepdims)))


def reduce_min(input_tensor, axis=None, keepdims=False, name=None):
    x = convert_to_tensor(input_tensor)
    return Tensor(_tf().min(x._js, _axes(axis), builtins.bool(keepdims)))


def softmax(logits, axis=-1, name=None):
    x = convert_to_tensor(logits)
    return Tensor(_tf().softmax(x._js, axis))


def one_hot(indices, depth, on_value=1, off_value=0, axis=-1, dtype=float32, name=None):
    if axis != -1:
        raise NotImplementedError("tf.one_hot currently supports axis=-1")
    x = cast(indices, int32)
    result = Tensor(_tf().oneHot(x._js, int(depth), on_value, off_value))
    if dtype != float32:
        return cast(result, dtype)
    return result


def zeros(shape, dtype=float32, name=None):
    return Tensor(_tf().zeros(to_js(list(shape)), dtype))


def ones(shape, dtype=float32, name=None):
    return Tensor(_tf().ones(to_js(list(shape)), dtype))


def zeros_like(input, dtype=None, name=None):
    x = convert_to_tensor(input)
    out = Tensor(_tf().zerosLike(x._js))
    return cast(out, dtype) if dtype and dtype != out.dtype else out


def ones_like(input, dtype=None, name=None):
    x = convert_to_tensor(input)
    out = Tensor(_tf().onesLike(x._js))
    return cast(out, dtype) if dtype and dtype != out.dtype else out


def eye(num_rows, num_columns=None, batch_shape=None, dtype=float32, name=None):
    if batch_shape is not None:
        raise NotImplementedError("batched eye is not implemented")
    return Tensor(_tf().eye(int(num_rows), int(num_columns or num_rows), None, dtype))


def concat(values, axis, name=None):
    js_values = [convert_to_tensor(v)._js for v in values]
    return Tensor(_tf().concat(globalThis.Array.of(*js_values), int(axis)))


def stack(values, axis=0, name=None):
    js_values = [convert_to_tensor(v)._js for v in values]
    return Tensor(_tf().stack(globalThis.Array.of(*js_values), int(axis)))


def clip_by_value(t, clip_value_min, clip_value_max, name=None):
    x = convert_to_tensor(t)
    return Tensor(_tf().clipByValue(x._js, clip_value_min, clip_value_max))


def random_normal(shape, mean=0.0, stddev=1.0, dtype=float32, seed=None, name=None):
    if seed is not None:
        raise NotImplementedError("TF.js randomNormal seed parity is not guaranteed")
    return Tensor(_tf().randomNormal(to_js(list(shape)), mean, stddev, dtype))


def random_uniform(shape, minval=0, maxval=None, dtype=float32, seed=None, name=None):
    if seed is not None:
        raise NotImplementedError("TF.js randomUniform seed parity is not guaranteed")
    if maxval is None:
        maxval = 1
    return Tensor(_tf().randomUniform(to_js(list(shape)), minval, maxval, dtype))


def grad(function):
    """Differentiate a scalar-valued eager Tensor function using TF.js tape.

    Callback is evaluated synchronously under tf.grad. No NumPy conversion is
    allowed inside it, as that would detach the computation graph.
    """
    def derivative(x):
        original = convert_to_tensor(x)
        retained = []
        def evaluate(js_value):
            arg = Tensor(js_value, owned=False)
            result = function(arg)
            if not isinstance(result, Tensor):
                raise TypeError("Differentiated function must return tf.Tensor")
            retained.append(result)
            if result.size != 1:
                raise ValueError("tf.grad requires a scalar loss")
            return result._js
        callback = create_proxy(evaluate)
        try:
            derivative_fn = _tf().grad(callback)
            try:
                result = derivative_fn(original._js)
                return Tensor(result)
            finally:
                retained.clear()
        finally:
            callback.destroy()
    return derivative
