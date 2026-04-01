#include "synara/ops/reduction.hpp"

namespace synara {

Tensor sum(const Tensor& a) {
    Tensor::value_type total = 0;

    for (Size i = 0; i < a.numel(); ++i) {
        total += a.data()[i];
    }

    return Tensor::from_vector(Shape({}), {total});
}


Tensor mean(const Tensor& a) {
    Tensor::value_type total = 0;


    for (Size i = 0; i < a.numel(); ++i) {
        total += a.data()[i];
    }

    return Tensor::from_vector(Shape({}), {total / static_cast<Tensor::value_type>(a.numel())});
}

} // namespace synara