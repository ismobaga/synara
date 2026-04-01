#include <cassert>
#include "synara/ops/loss.hpp"
#include "synara/tensor/tensor.hpp"

int main()
{
    using namespace synara;
    Tensor pred = Tensor::from_vector(Shape({2, 3}), {1.0f, 2.0f, 3.0f,
                                                      4.0f, 5.0f, 6.0f});
    Tensor target = Tensor::from_vector(Shape({2, 3}), {1.0f, 2.0f, 3.0f,
                                                        4.0f, 5.0f, 6.0f});
    Tensor loss = mse_loss(pred, target);
    assert(loss.is_scalar());
    assert(loss.item() == 0.0f);

    // ((2-1)^2 + (4-1)^2) / 2 = (1 + 9) / 2 = 5
    Tensor pred2 = Tensor::from_vector({2, 1}, {2, 4});
    Tensor target2 = Tensor::from_vector({2, 1}, {1, 1});
    Tensor loss2 = mse_loss(pred2, target2);
    assert(loss2.is_scalar());
    assert(loss2.item() == 5.0f);

    return 0;
}
