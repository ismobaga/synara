#include <cassert>
#include <cmath>
#include <algorithm>

#include "synara/tensor/tensor.hpp"

int main()
{
    using namespace synara;

    Tensor t = Tensor::randn(Shape({100, 100}));
    assert(t.shape() == Shape({100, 100}));
    assert(t.numel() == 10000);

    Tensor::value_type sum = 0.0;
    for (std::size_t i = 0; i < t.numel(); ++i)
    {
        sum += t.data()[i];
    }
    Tensor::value_type mean = sum / static_cast<Tensor::value_type>(t.numel());

    Tensor::value_type sq_sum = 0.0;
    for (std::size_t i = 0; i < t.numel(); ++i)
    {
        Tensor::value_type diff = t.data()[i] - mean;
        sq_sum += diff * diff;
    }
    Tensor::value_type var = sq_sum / static_cast<Tensor::value_type>(t.numel());
    Tensor::value_type stddev = std::sqrt(var);

    assert(std::fabs(mean) < 0.2);
    assert(std::fabs(stddev - 1.0) < 0.2);

    return 0;
}
