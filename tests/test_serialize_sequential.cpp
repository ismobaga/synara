#include <cassert>
#include <cstdio>
#include <cmath>
#include <memory>

#include "synara/nn/linear.hpp"
#include "synara/nn/relu.hpp"
#include "synara/nn/sequential.hpp"
#include "synara/nn/sigmoid.hpp"
#include "synara/serialize/state_dict.hpp"

int main()
{
    using namespace synara;

    auto l1 = std::make_shared<Linear>(2, 3, true);
    auto a1 = std::make_shared<ReLU>();
    auto l2 = std::make_shared<Linear>(3, 1, true);
    auto a2 = std::make_shared<Sigmoid>();

    Sequential src({l1, a1, l2, a2});

    l1->weight().tensor().at({0, 0}) = 0.5f;
    l1->weight().tensor().at({0, 1}) = -0.1f;
    l1->weight().tensor().at({1, 0}) = 0.2f;
    l1->weight().tensor().at({1, 1}) = 0.3f;
    l1->weight().tensor().at({2, 0}) = -0.4f;
    l1->weight().tensor().at({2, 1}) = 0.7f;
    l1->bias().tensor().at({0, 0}) = 0.01f;
    l1->bias().tensor().at({0, 1}) = 0.02f;
    l1->bias().tensor().at({0, 2}) = 0.03f;

    l2->weight().tensor().at({0, 0}) = 0.8f;
    l2->weight().tensor().at({0, 1}) = -0.6f;
    l2->weight().tensor().at({0, 2}) = 0.4f;
    l2->bias().tensor().at({0, 0}) = -0.2f;

    const std::string path = "seq_state.ckpt";
    save_state_dict(src.state_dict(), path);

    auto l1b = std::make_shared<Linear>(2, 3, true);
    auto a1b = std::make_shared<ReLU>();
    auto l2b = std::make_shared<Linear>(3, 1, true);
    auto a2b = std::make_shared<Sigmoid>();

    Sequential dst({l1b, a1b, l2b, a2b});
    StateDict loaded = load_state_dict(path);
    dst.load_state_dict(loaded);

    Tensor x = Tensor::from_vector(Shape({2, 2}), {0.0f, 1.0f, 1.0f, 1.0f});
    Tensor y1 = src(x);
    Tensor y2 = dst(x);

    assert(y1.shape() == y2.shape());
    for (std::size_t i = 0; i < y1.numel(); ++i)
    {
        assert(std::fabs(y1.data()[i] - y2.data()[i]) < 1e-6f);
    }

    std::remove(path.c_str());
    return 0;
}
