#include "synara/core/parallel.hpp"

#include <algorithm>

#if defined(SYNARA_USE_OPENMP)
#include <omp.h>
#endif

namespace synara
{
    namespace
    {

        ParallelConfig &global_parallel_config()
        {
            static ParallelConfig config{};
            return config;
        }

    } // namespace

    bool openmp_enabled() noexcept
    {
#if defined(SYNARA_USE_OPENMP)
        return true;
#else
        return false;
#endif
    }

    int get_num_threads() noexcept
    {
#if defined(SYNARA_USE_OPENMP)
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

    void set_num_threads(int num_threads) noexcept
    {
        if (num_threads < 1)
        {
            num_threads = 1;
        }

#if defined(SYNARA_USE_OPENMP)
        omp_set_num_threads(num_threads);
#else
        (void)num_threads;
#endif
    }

    const ParallelConfig &parallel_config() noexcept
    {
        return global_parallel_config();
    }

    void set_parallel_config(const ParallelConfig &config) noexcept
    {
        global_parallel_config() = config;
    }

    void reset_parallel_config() noexcept
    {
        global_parallel_config() = ParallelConfig{};
    }

    ParallelConfig autotune_parallel_config(Size workload_size, int preferred_threads) noexcept
    {
        ParallelConfig config{};

        int threads = preferred_threads > 0 ? preferred_threads : get_num_threads();
        if (threads < 1)
        {
            threads = 1;
        }

        if (workload_size <= static_cast<Size>(1) << 12)
        {
            config.elementwise_threshold = static_cast<Size>(1) << 20;
            config.matmul_threshold = static_cast<Size>(1) << 18;
            config.linear_threshold = static_cast<Size>(1) << 18;
            config.conv2d_threshold = static_cast<Size>(1) << 18;
            config.pooling_threshold = static_cast<Size>(1) << 18;
            return config;
        }

        const Size thread_scale = static_cast<Size>(threads);
        config.elementwise_threshold = std::max<Size>((static_cast<Size>(1) << 14) * thread_scale, workload_size / 4);
        config.matmul_threshold = std::max<Size>((static_cast<Size>(1) << 13) * thread_scale, workload_size / 8);
        config.linear_threshold = std::max<Size>((static_cast<Size>(1) << 13) * thread_scale, workload_size / 8);
        config.conv2d_threshold = std::max<Size>((static_cast<Size>(1) << 12) * thread_scale, workload_size / 16);
        config.pooling_threshold = std::max<Size>((static_cast<Size>(1) << 12) * thread_scale, workload_size / 16);
        return config;
    }

    void apply_autotuned_parallel_config(Size workload_size, int preferred_threads) noexcept
    {
        if (preferred_threads > 0)
        {
            set_num_threads(preferred_threads);
        }
        set_parallel_config(autotune_parallel_config(workload_size, preferred_threads));
    }

} // namespace synara
