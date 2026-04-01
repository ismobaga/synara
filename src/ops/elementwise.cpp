#include "synara/ops/elementwise.hpp"
#include "synara/autograd/nodes.hpp"
#include <cmath>

#include "synara/core/error.hpp"

namespace synara {

static void require_same_shape(const Tensor& a, const Tensor& b, const char* op_name) {
    if (a.shape() != b.shape()) {
        throw ShapeError(std::string(op_name) + ": tensors must have the same shape.");
    }
}

static void require_contiguous(const Tensor& t, const char* op_name) {
    if (!t.is_contiguous()) {
        throw ShapeError(std::string(op_name) + ": tensors must be contiguous.");
    }
}

bool compute_requires_grad(const Tensor& a, const Tensor& b) {
    return a.requires_grad() || b.requires_grad();
}

bool compute_requires_grad(const Tensor& a) {
    return a.requires_grad();
}


static Tensor unary_scalar_op(
    const Tensor& a,
    Tensor::value_type scalar,
    Tensor::value_type (*fn)(Tensor::value_type, Tensor::value_type)
) {
    require_contiguous(a, "unary scalar op");
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
    require_contiguous(a, "elementwise op");
    require_contiguous(b, "elementwise op");

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
    Tensor out =  binary_tensor_op(a, b, add_fn);
    bool req = compute_requires_grad(a, b);
    out.set_leaf(!req);
    out.set_requires_grad(req);
    if (req) {
        auto node = std::make_shared<AddNode>(a, b);
        out.set_grad_fn(node);
    }
    return out;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b, "sub");
    Tensor out = binary_tensor_op(a, b, sub_fn);
    bool req = compute_requires_grad(a, b);
    out.set_leaf(!req);
    out.set_requires_grad(req);
    if (req) {
        auto node = std::make_shared<SubNode>(a, b);
        out.set_grad_fn(node);
    }
    return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b, "mul");
    Tensor out = binary_tensor_op(a, b, mul_fn);
    bool req = compute_requires_grad(a, b);
    out.set_leaf(!req);
    out.set_requires_grad(req);

    if (req) {
        auto node = std::make_shared<MulNode>(a, b);
        out.set_grad_fn(node);
    }
    return out;
}

Tensor div(const Tensor& a, const Tensor& b) {
    require_same_shape(a, b, "div");
    Tensor out = binary_tensor_op(a, b, div_fn);
    bool req = compute_requires_grad(a, b);
    out.set_leaf(!req);
    out.set_requires_grad(req);
    if (req) {
        auto node = std::make_shared<DivNode>(a, b);
        out.set_grad_fn(node);
    }
    return out;
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