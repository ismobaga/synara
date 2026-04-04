#include <cassert>

#include "synara/core/parallel.hpp"

int main()
{
    using namespace synara;

    assert(get_num_threads() >= 1);

    const ParallelConfig defaults = parallel_config();
    ParallelConfig tuned = defaults;
    tuned.elementwise_threshold = 1024;
    tuned.matmul_threshold = 2048;
    tuned.linear_threshold = 4096;
    tuned.conv2d_threshold = 8192;
    tuned.pooling_threshold = 16384;

    set_parallel_config(tuned);
    const ParallelConfig &current = parallel_config();
    assert(current.elementwise_threshold == 1024);
    assert(current.matmul_threshold == 2048);
    assert(current.linear_threshold == 4096);
    assert(current.conv2d_threshold == 8192);
    assert(current.pooling_threshold == 16384);

    const int previous_threads = get_num_threads();
    set_num_threads(1);
    assert(get_num_threads() >= 1);
    set_num_threads(previous_threads);

    reset_parallel_config();
    const ParallelConfig &reset = parallel_config();
    assert(reset.elementwise_threshold == ParallelConfig{}.elementwise_threshold);
    assert(reset.matmul_threshold == ParallelConfig{}.matmul_threshold);
    assert(reset.linear_threshold == ParallelConfig{}.linear_threshold);
    assert(reset.conv2d_threshold == ParallelConfig{}.conv2d_threshold);
    assert(reset.pooling_threshold == ParallelConfig{}.pooling_threshold);

    const ParallelConfig auto_cfg = autotune_parallel_config(1 << 18, 4);
    assert(auto_cfg.elementwise_threshold > 0);
    assert(auto_cfg.matmul_threshold > 0);
    assert(auto_cfg.linear_threshold > 0);
    assert(auto_cfg.conv2d_threshold > 0);
    assert(auto_cfg.pooling_threshold > 0);

    apply_autotuned_parallel_config(1 << 18, 2);
    const ParallelConfig &applied = parallel_config();
    assert(applied.elementwise_threshold > 0);
    assert(applied.conv2d_threshold > 0);
    assert(get_num_threads() >= 1);

    return 0;
}
