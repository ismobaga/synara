#include "synara/tensor/tensor.hpp"

namespace synara {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

Tensor add(const Tensor& a, Tensor::value_type scalar);
Tensor sub(const Tensor& a, Tensor::value_type scalar);
Tensor mul(const Tensor& a, Tensor::value_type scalar);
Tensor div(const Tensor& a, Tensor::value_type scalar);

Tensor add(Tensor::value_type scalar, const Tensor& a);
Tensor sub(Tensor::value_type scalar, const Tensor& a);
Tensor mul(Tensor::value_type scalar, const Tensor& a);
Tensor div(Tensor::value_type scalar, const Tensor& a);

} // namespace synara