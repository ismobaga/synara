#include <cassert>
#include <chrono>
#include <thread>

#include "synara/core/profiler.hpp"

int main()
{
    using namespace synara;

    reset_profile_data();
    enable_profiling(true);

    {
        ScopedProfile profile("sleep_scope");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    const ProfileStats stats = get_profile_stats("sleep_scope");
    assert(stats.name == "sleep_scope");
    assert(stats.calls == 1);
    assert(stats.total_ms >= 0.0);
    assert(stats.max_ms >= stats.min_ms);
    assert(stats.average_ms() >= 0.0);

    const std::string report = format_profile_summary();
    assert(report.find("sleep_scope") != std::string::npos);

    enable_profiling(false);
    {
        ScopedProfile disabled_profile("disabled_scope");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    const ProfileStats disabled = get_profile_stats("disabled_scope");
    assert(disabled.calls == 0);

    enable_profiling(true);
    reset_profile_data();
    assert(profile_summary().empty());

    return 0;
}
