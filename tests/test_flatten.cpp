#include <cassert>
#include <cmath>

#include "synara/ops/elementwise.hpp"
#include "synara/ops/reduction.hpp"
#include "synara/tensor/tensor.hpp"

using namespace synara;

namespace
{

void test_flatten_basic()
{
    Tensor t({2, 3}, 1.0f);
    Tensor flat = t.flatten();
    assert(flat.shape() == Shape({6}));
    for (std::size_t i = 0; i < flat.numel(); ++i)
        assert(flat.data()[i] == 1.0f);
}

void test_flatten_1d()
{
    // Flattening a 1-D tensor is a no-op (same shape)
    Tensor t = Tensor::from_vector(Shape({5}), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor flat = t.flatten();
    assert(flat.shape() == Shape({5}));
    assert(flat.numel() == 5);
}

void test_flatten_4d()
{
    // (2, 3, 4, 5) => numel = 120
    Tensor t(Shape({2, 3, 4, 5}), 0.5f);
    Tensor flat = t.flatten();
    assert(flat.shape() == Shape({120}));
    for (std::size_t i = 0; i < flat.numel(); ++i)
        assert(flat.data()[i] == 0.5f);
}

void test_flatten_scalar()
{
    // A scalar tensor (rank-0) should flatten to shape {1}
    Tensor t = Tensor::from_vector(Shape({}), {3.14f});
    Tensor flat = t.flatten();
    assert(flat.numel() == 1);
    assert(flat.data()[0] == 3.14f);
}

void test_flatten_gradient_flow()
{
    // gradient of sum(flatten(x)) w.r.t. x should be all-ones with the
    // same shape as x.
    auto x    = Tensor::from_vector(Shape({2, 3}),
                                    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                                    true);
    auto flat = x.flatten();
    auto s    = sum(flat);
    s.backward();

    assert(x.has_grad());
    assert(x.grad().shape() == x.shape());
    for (std::size_t i = 0; i < x.numel(); ++i)
        assert(x.grad().data()[i] == 1.0f);
}

void test_flatten_gradient_through_elementwise()
{
    // y = 2 * x  (elementwise), then flatten, then sum -> backward
    // dL/dx[i] should be 2 for every element.
    auto x = Tensor::from_vector(Shape({2, 4}),
                                 {1.0f, 2.0f, 3.0f, 4.0f,
                                  5.0f, 6.0f, 7.0f, 8.0f},
                                 true);
    auto two = Tensor::from_vector(Shape({2, 4}),
                                   {2.0f, 2.0f, 2.0f, 2.0f,
                                    2.0f, 2.0f, 2.0f, 2.0f});
    auto y    = mul(x, two);
    auto flat = y.flatten();
    auto s    = sum(flat);
    s.backward();

    assert(x.has_grad());
    for (std::size_t i = 0; i < x.numel(); ++i)
        assert(x.grad().data()[i] == 2.0f);
}

} // namespace

int main()
{
    test_flatten_basic();
    test_flatten_1d();
    test_flatten_4d();
    test_flatten_scalar();
    test_flatten_gradient_flow();
    test_flatten_gradient_through_elementwise();
    return 0;
}
