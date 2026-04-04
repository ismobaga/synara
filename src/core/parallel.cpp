#include "synara/core/parallel.hpp"

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

} // namespace synara
