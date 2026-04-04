#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "synara/core/types.hpp"

namespace synara
{

    struct ProfileStats
    {
        std::string name;
        Size calls = 0;
        double total_ms = 0.0;
        double min_ms = 0.0;
        double max_ms = 0.0;

        double average_ms() const noexcept
        {
            return calls == 0 ? 0.0 : total_ms / static_cast<double>(calls);
        }
    };

    bool profiling_enabled() noexcept;
    void enable_profiling(bool enabled) noexcept;
    void reset_profile_data() noexcept;

    void record_profile_event(const std::string &name, double milliseconds);
    ProfileStats get_profile_stats(const std::string &name);
    std::vector<ProfileStats> profile_summary();
    std::string format_profile_summary();
    std::string format_profile_csv();
    std::string format_profile_json();
    bool write_profile_summary(const std::string &path);
    bool write_profile_csv(const std::string &path);
    bool write_profile_json(const std::string &path);

    class ScopedProfile
    {
    public:
        explicit ScopedProfile(std::string name, bool active = true) noexcept;
        ~ScopedProfile();

        void stop() noexcept;
        double elapsed_ms() const noexcept;

    private:
        using clock = std::chrono::steady_clock;

        std::string name_;
        bool active_;
        bool stopped_;
        clock::time_point start_;
        double recorded_ms_;
    };

} // namespace synara
