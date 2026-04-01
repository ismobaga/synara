#include <cassert>
#include <memory>

#include "synara/nn/linear.hpp"
#include "synara/nn/sequential.hpp"

int main()
{
    using namespace synara;

    auto first = std::make_shared<Linear>(2, 3, true);
    auto second = std::make_shared<Linear>(3, 1, true);

    first->weight().tensor().at({0, 0}) = 1.0f;
    first->weight().tensor().at({0, 1}) = 0.0f;
    first->weight().tensor().at({1, 0}) = 0.0f;
    first->weight().tensor().at({1, 1}) = 1.0f;
    first->weight().tensor().at({2, 0}) = 1.0f;
    first->weight().tensor().at({2, 1}) = 1.0f;
    first->bias().tensor().at({0, 0}) = 0.0f;
    first->bias().tensor().at({0, 1}) = 1.0f;
    first->bias().tensor().at({0, 2}) = -1.0f;

    second->weight().tensor().at({0, 0}) = 2.0f;
    second->weight().tensor().at({0, 1}) = -1.0f;
    second->weight().tensor().at({0, 2}) = 0.5f;
    second->bias().tensor().at({0, 0}) = 3.0f;

    Sequential model;
    model.add(first);
    model.add(second);

    Tensor input = Tensor::from_vector(Shape({1, 2}), {2.0f, 4.0f});
    Tensor output = model(input);

    assert(model.size() == 2);
    assert(model.parameters().size() == 4);
    assert(output.shape() == Shape({1, 1}));
    assert(output.at({0, 0}) == 4.5f);

    return 0;
}