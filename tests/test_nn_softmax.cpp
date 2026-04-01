#include <cassert>
#include <cmath>

#include "synara/nn/softmax.hpp"

int main()
{
    using namespace synara;

    Softmax layer(1);
    Tensor x = Tensor::from_vector(Shape({2, 3}), {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    });
    Tensor y = layer(x);

    constexpr float tol = 1e-6f;
    assert(y.shape() == Shape({2, 3}));

    for (size_t i = 0; i < 2; ++i)
    {
        float row_sum = y.at({i, 0}) + y.at({i, 1}) + y.at({i, 2});
        assert(std::fabs(row_sum - 1.0f) < tol);

        for (size_t j = 0; j < 3; ++j)
        {
            assert(y.at({i, j}) >= 0.0f && y.at({i, j}) <= 1.0f);
        }
    }

    return 0;
}
