#pragma once

#include "synara/core/types.hpp"

namespace synara
{

    struct ParallelConfig
    {
        Size elementwise_threshold = static_cast<Size>(1) << 18;
        Size matmul_threshold = static_cast<Size>(1) << 15;
        Size linear_threshold = static_cast<Size>(1) << 16;
        Size conv2d_threshold = static_cast<Size>(1) << 16;
        Size pooling_threshold = static_cast<Size>(1) << 15;
    };

    bool openmp_enabled() noexcept;
    int get_num_threads() noexcept;
    void set_num_threads(int num_threads) noexcept;

    const ParallelConfig &parallel_config() noexcept;
    void set_parallel_config(const ParallelConfig &config) noexcept;
    void reset_parallel_config() noexcept;

    // Heuristic tuning based on a representative workload size.
    ParallelConfig autotune_parallel_config(Size workload_size,
                                            int preferred_threads = 0) noexcept;
    void apply_autotuned_parallel_config(Size workload_size,
                                         int preferred_threads = 0) noexcept;

} // namespace synara
