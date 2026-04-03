#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#include "synara/data/dataloader.hpp"

namespace
{

    void test_num_batches_ceil_division()
    {
        using namespace synara;

        auto x = Tensor::from_vector(Shape({5, 2}), {
                                                        1,
                                                        2,
                                                        3,
                                                        4,
                                                        5,
                                                        6,
                                                        7,
                                                        8,
                                                        9,
                                                        10,
                                                    });
        auto y = Tensor::from_vector(Shape({5, 1}), {0, 1, 0, 1, 0});

        TensorDataset ds(x, y);
        DataLoader dl(ds, 2, false, 123);
        assert(dl.num_batches() == 3);

        std::size_t seen = 0;
        std::vector<std::size_t> batch_sizes;
        for (auto batch : dl)
        {
            const auto &bx = batch.first;
            const auto &by = batch.second;
            batch_sizes.push_back(bx.shape()[0]);
            assert(bx.shape()[0] == by.shape()[0]);
            seen += bx.shape()[0];
        }

        assert(seen == 5);
        assert(batch_sizes.size() == 3);
        assert(batch_sizes[0] == 2);
        assert(batch_sizes[1] == 2);
        assert(batch_sizes[2] == 1);
    }

    void test_shuffle_changes_order()
    {
        using namespace synara;

        auto x = Tensor::from_vector(Shape({6, 1}), {0, 1, 2, 3, 4, 5});
        auto y = Tensor::from_vector(Shape({6, 1}), {0, 0, 0, 0, 0, 0});

        TensorDataset ds(x, y);
        DataLoader dl_a(ds, 3, true, 7);
        DataLoader dl_b(ds, 3, true, 8);

        std::vector<float> order_a;
        std::vector<float> order_b;

        for (auto batch : dl_a)
        {
            auto bx = batch.first.contiguous();
            for (std::size_t i = 0; i < bx.shape()[0]; ++i)
            {
                order_a.push_back(bx.at({i, 0}));
            }
        }

        for (auto batch : dl_b)
        {
            auto bx = batch.first.contiguous();
            for (std::size_t i = 0; i < bx.shape()[0]; ++i)
            {
                order_b.push_back(bx.at({i, 0}));
            }
        }

        assert(order_a.size() == 6);
        assert(order_b.size() == 6);
        assert(order_a != order_b);
    }

} // namespace

int main()
{
    test_num_batches_ceil_division();
    test_shuffle_changes_order();
    return 0;
}
