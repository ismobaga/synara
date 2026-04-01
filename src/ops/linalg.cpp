#include "synara/ops/linalg.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/core/error.hpp"

namespace synara {

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.rank() != 2 || b.rank() != 2) {
        throw ShapeError("matmul(): only rank-2 tensors are supported.");
    }

    const Size m = a.shape()[0];
    const Size k1 = a.shape()[1];
    const Size k2 = b.shape()[0];
    const Size n = b.shape()[1];

    if (k1 != k2) {
        throw ShapeError("matmul(): inner dimensions must match.");
    }

    Tensor out = Tensor::zeros(Shape({m, n}));

    for (Size i = 0; i < m; ++i) {
        for (Size j = 0; j < n; ++j) {
            Tensor::value_type acc = 0;
            for (Size k = 0; k < k1; ++k) {
                acc += a.at({i, k}) * b.at({k, j});
            }
            out.at({i, j}) = acc;
        }
    }

    bool req = a.requires_grad() || b.requires_grad();
    out.set_leaf(!req);
    out.set_requires_grad(req);
    if (req) {
        auto node = std::make_shared<MatMulNode>(a, b);
        out.set_grad_fn(node);
    }

    return out;
}

} // namespace synara