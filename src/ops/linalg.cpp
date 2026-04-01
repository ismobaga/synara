#include "synara/ops/linalg.hpp"

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

    return out;
}

} // namespace synara