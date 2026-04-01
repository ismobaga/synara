#include <cassert>
#include <cstdio>
#include <cmath>

#include "synara/nn/conv2d.hpp"
#include "synara/serialize/state_dict.hpp"

int main()
{
    using namespace synara;

    Conv2d src(1, 2, 2, 2, 1, 1, 1, 1, true);

    src.weight().tensor().at({0, 0, 0, 0}) = 0.1f;
    src.weight().tensor().at({0, 0, 0, 1}) = 0.2f;
    src.weight().tensor().at({0, 0, 1, 0}) = 0.3f;
    src.weight().tensor().at({0, 0, 1, 1}) = 0.4f;

    src.weight().tensor().at({1, 0, 0, 0}) = -0.1f;
    src.weight().tensor().at({1, 0, 0, 1}) = -0.2f;
    src.weight().tensor().at({1, 0, 1, 0}) = -0.3f;
    src.weight().tensor().at({1, 0, 1, 1}) = -0.4f;

    src.bias().tensor().at({0}) = 0.5f;
    src.bias().tensor().at({1}) = -0.5f;

    const std::string path = "conv2d_state.ckpt";
    save_state_dict(src.state_dict(), path);

    Conv2d dst(1, 2, 2, 2, 1, 1, 1, 1, true);
    StateDict loaded = load_state_dict(path);
    dst.load_state_dict(loaded);

    Tensor x = Tensor::from_vector(
        Shape({1, 1, 3, 3}),
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
        });

    Tensor y1 = src(x);
    Tensor y2 = dst(x);

    assert(y1.shape() == y2.shape());
    for (Size i = 0; i < y1.numel(); ++i)
    {
        assert(std::fabs(y1.data()[i] - y2.data()[i]) < 1e-6f);
    }

    std::remove(path.c_str());
    return 0;
}
