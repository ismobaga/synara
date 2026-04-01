#include "synara/ops/activation.hpp"

namespace synara {

Tensor relu(const Tensor& a) {
    Tensor out = Tensor::zeros(a.shape());

    for (Size i = 0; i < a.numel(); ++i) {
        out.data()[i] = (a.data()[i] > 0.0f) ? a.data()[i] : 0.0f;
    }

    return out;
}

} // namespace synara