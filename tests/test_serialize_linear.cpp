#include <cassert>
#include <cstdio>

#include "synara/nn/linear.hpp"
#include "synara/serialize/state_dict.hpp"

int main()
{
    using namespace synara;

    Linear src(2, 1, true);
    src.weight().tensor().at({0, 0}) = 1.5f;
    src.weight().tensor().at({0, 1}) = -0.75f;
    src.bias().tensor().at({0, 0}) = 0.25f;

    const std::string path = "linear_state.ckpt";
    save_state_dict(src.state_dict(), path);

    Linear dst(2, 1, true);
    StateDict loaded = load_state_dict(path);
    dst.load_state_dict(loaded);

    Tensor x = Tensor::from_vector(Shape({3, 2}), {1.0f, 2.0f, 0.0f, -1.0f, 4.0f, 3.0f});
    Tensor y1 = src(x);
    Tensor y2 = dst(x);

    assert(y1.shape() == y2.shape());
    for (std::size_t i = 0; i < y1.numel(); ++i)
    {
        assert(y1.data()[i] == y2.data()[i]);
    }

    std::remove(path.c_str());
    return 0;
}
