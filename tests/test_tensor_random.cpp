#include <cassert>
#include <cmath>

#include "synara/tensor/tensor.hpp"

int main()
{
    using namespace synara;

    {
        Tensor t = Tensor::uniform(Shape({128, 16}), -2.0, 3.0, true);
        assert(t.shape() == Shape({128, 16}));
        assert(t.requires_grad());
        for (Size i = 0; i < t.numel(); ++i)
        {
            assert(t.data()[i] >= -2.0);
            assert(t.data()[i] <= 3.0);
        }
    }

    {
        Tensor t = Tensor::randn(Shape({10000}), 1.5, 0.5, false);
        assert(t.shape() == Shape({10000}));
        assert(!t.requires_grad());

        double mean = 0.0;
        for (Size i = 0; i < t.numel(); ++i)
        {
            mean += t.data()[i];
        }
        mean /= static_cast<double>(t.numel());

        double var = 0.0;
        for (Size i = 0; i < t.numel(); ++i)
        {
            const double d = t.data()[i] - mean;
            var += d * d;
        }
        var /= static_cast<double>(t.numel());
        const double stddev = std::sqrt(var);

        assert(std::fabs(mean - 1.5) < 0.08);
        assert(std::fabs(stddev - 0.5) < 0.08);
    }

    {
        bool threw = false;
        try
        {
            (void)Tensor::randn(Shape({2, 2}), 0.0, 0.0, false);
        }
        catch (const ValueError &)
        {
            threw = true;
        }
        assert(threw);
    }

    {
        bool threw = false;
        try
        {
            (void)Tensor::uniform(Shape({2, 2}), 5.0, -1.0, false);
        }
        catch (const ValueError &)
        {
            threw = true;
        }
        assert(threw);
    }

    return 0;
}
