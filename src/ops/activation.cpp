#include "synara/ops/activation.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/core/error.hpp"

namespace synara {

Tensor relu(const Tensor& a) {
    Tensor out = Tensor::zeros(a.shape());

    for (Size i = 0; i < a.numel(); ++i) {
        out.data()[i] = (a.data()[i] > 0.0f) ? a.data()[i] : 0.0f;
    }

    out.set_leaf(!a.requires_grad());
    out.set_requires_grad(a.requires_grad());
    if (a.requires_grad()) {
        auto node = std::make_shared<ReLUNode>(a);
        out.set_grad_fn(node);
    }

    return out;
}

} // namespace synara