#include "synara/ops/reduction.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/core/error.hpp"

namespace synara {

    static void require_contiguous(const Tensor& t, const char* op_name) {
    if (!t.is_contiguous()) {
        throw ValueError(std::string(op_name) + ": requires contiguous tensor in current milestone.");
    }
}

Tensor sum(const Tensor& a) {
    require_contiguous(a, "sum");
    Tensor::value_type total = 0;

    for (Size i = 0; i < a.numel(); ++i) {
        total += a.data()[i];
    }

    Tensor out = Tensor::from_vector(Shape({}), {total}, a.requires_grad());
    out.set_leaf(!a.requires_grad());
    if (a.requires_grad()) {
        auto node = std::make_shared<SumNode>(a);
        out.set_grad_fn(node);
    }
    return out;
}


Tensor mean(const Tensor& a) {
    require_contiguous(a, "mean");
    Tensor::value_type total = 0;


    for (Size i = 0; i < a.numel(); ++i) {
        total += a.data()[i];
    }

    Tensor out = Tensor::from_vector(Shape({}), {total / static_cast<Tensor::value_type>(a.numel())}, a.requires_grad());
    out.set_leaf(!a.requires_grad());
    if (a.requires_grad()) {
        auto node = std::make_shared<MeanNode>(a);
        out.set_grad_fn(node);
    }
    return out;
}

} // namespace synara