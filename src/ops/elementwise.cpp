#include "synara/ops/elementwise.hpp"

#include <cmath>

#include "synara/core/error.hpp"

namespace synara {

static void require_same_shape(const Tensor& a, const Tensor& b, const char* op_name) {
    if (a.shape() != b.shape()) {
        throw ShapeError(std::string(op_name) + ": tensors must have the same shape.");
    }
}

static Tensor unary_scalar_op(
    const Tensor& a,
    Tensor::value_type scalar,
    Tensor::value_type (*fn)(Tensor::value_type, Tensor::value_type)
) {
    Tensor out = Tensor::zeros(a.shape());

    for (Size i = 0; i < a.numel(); ++i) {
        out.data()[i] = fn(a.data()[i], scalar);
    }

    return out;
}

static Tensor binary_tensor_op(
    const Tensor& a,
    const Tensor& b,
    Tensor::value_type (*fn)(Tensor::value_type, Tensor::value_type)
) {
    require_same_shape(a, b, "elementwise op");

    Tensor out = Tensor::zeros(a.shape());

    for (Size i = 0; i < a.numel(); ++i) {
        out.data()[i] = fn(a.data()[i], b.data()[i]);
    }

    return out;
}

static Tensor::value_type add_fn(Tensor::value_type x, Tensor::value_type y) { return x + y; }
static Tensor::value_type sub_fn(Tensor::value_type x, Tensor::value_type y) { return x - y; }
static Tensor::value_type mul_fn(Tensor::value_type x, Tensor::value_type y) { return x * y; }
static Tensor::value_type div_fn(Tensor::value_type x, Tensor::value_type y) { return x / y; }

Tensor add(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b, "add");
    return binary_tensor_op(a, b, add_fn);
}

Tensor sub(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b, "sub");
    return binary_tensor_op(a, b, sub_fn);
}

Tensor mul(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b, "mul");
    return binary_tensor_op(a, b, mul_fn);
}

Tensor div(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b, "div");
    return binary_tensor_op(a, b, div_fn);
}

Tensor add(const Tensor& a, Tensor::value_type scalar) {
    return unary_scalar_op(a, scalar, add_fn);
}

Tensor sub(const Tensor& a, Tensor::value_type scalar) {
    return unary_scalar_op(a, scalar, sub_fn);
}

Tensor mul(const Tensor& a, Tensor::value_type scalar) {
    return unary_scalar_op(a, scalar, mul_fn);
}

Tensor div(const Tensor& a, Tensor::value_type scalar) {
    return unary_scalar_op(a, scalar, div_fn);
}

Tensor add(Tensor::value_type scalar, const Tensor& a) {
    return add(a, scalar);
}

Tensor sub(Tensor::value_type scalar, const Tensor& a) {
    Tensor out = Tensor::zeros(a.shape());
    for (Size i = 0; i < a.numel(); ++i) {
        out.data()[i] = scalar - a.data()[i];
    }
    return out;
}

Tensor mul(Tensor::value_type scalar, const Tensor& a) {
    return mul(a, scalar);
}

Tensor div(Tensor::value_type scalar, const Tensor& a) {
    Tensor out = Tensor::zeros(a.shape());
    for (Size i = 0; i < a.numel(); ++i) {
        out.data()[i] = scalar / a.data()[i];
    }
    return out;
}

} // namespace synara