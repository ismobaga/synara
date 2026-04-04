#include "synara/core/profiler.hpp"

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace synara
{
    namespace
    {

        std::unordered_map<std::string, ProfileStats> &profile_storage()
        {
            static std::unordered_map<std::string, ProfileStats> stats;
            return stats;
        }

        std::mutex &profile_mutex()
        {
            static std::mutex mutex;
            return mutex;
        }

        std::atomic<bool> &profile_enabled_flag()
        {
            static std::atomic<bool> enabled{true};
            return enabled;
        }

        std::string escape_json(const std::string &value)
        {
            std::string escaped;
            escaped.reserve(value.size());
            for (const char c : value)
            {
                switch (c)
                {
                case '\\':
                    escaped += "\\\\";
                    break;
                case '"':
                    escaped += "\\\"";
                    break;
                case '\n':
                    escaped += "\\n";
                    break;
                case '\r':
                    escaped += "\\r";
                    break;
                case '\t':
                    escaped += "\\t";
                    break;
                default:
                    escaped.push_back(c);
                    break;
                }
            }
            return escaped;
        }

        std::string escape_csv(const std::string &value)
        {
            if (value.find_first_of(",\"\n\r") == std::string::npos)
            {
                return value;
            }

            std::string escaped = "\"";
            for (const char c : value)
            {
                if (c == '"')
                {
                    escaped += "\"\"";
                }
                else
                {
                    escaped.push_back(c);
                }
            }
            escaped.push_back('"');
            return escaped;
        }

    } // namespace

    bool profiling_enabled() noexcept
    {
        return profile_enabled_flag().load(std::memory_order_relaxed);
    }

    void enable_profiling(bool enabled) noexcept
    {
        profile_enabled_flag().store(enabled, std::memory_order_relaxed);
    }

    void reset_profile_data() noexcept
    {
        std::lock_guard<std::mutex> lock(profile_mutex());
        profile_storage().clear();
    }

    void record_profile_event(const std::string &name, double milliseconds)
    {
        if (!profiling_enabled() || name.empty())
        {
            return;
        }

        std::lock_guard<std::mutex> lock(profile_mutex());
        ProfileStats &stats = profile_storage()[name];
        if (stats.calls == 0)
        {
            stats.name = name;
            stats.min_ms = milliseconds;
            stats.max_ms = milliseconds;
        }
        else
        {
            stats.min_ms = std::min(stats.min_ms, milliseconds);
            stats.max_ms = std::max(stats.max_ms, milliseconds);
        }

        ++stats.calls;
        stats.total_ms += milliseconds;
    }

    ProfileStats get_profile_stats(const std::string &name)
    {
        std::lock_guard<std::mutex> lock(profile_mutex());
        const auto it = profile_storage().find(name);
        if (it != profile_storage().end())
        {
            return it->second;
        }

        ProfileStats empty;
        empty.name = name;
        return empty;
    }

    std::vector<ProfileStats> profile_summary()
    {
        std::lock_guard<std::mutex> lock(profile_mutex());
        std::vector<ProfileStats> summary;
        summary.reserve(profile_storage().size());
        for (const auto &entry : profile_storage())
        {
            summary.push_back(entry.second);
        }

        std::sort(summary.begin(), summary.end(), [](const ProfileStats &lhs, const ProfileStats &rhs)
                  { return lhs.total_ms > rhs.total_ms; });
        return summary;
    }

    std::string format_profile_summary()
    {
        const auto summary = profile_summary();
        std::ostringstream oss;
        oss << "Profiler summary\n";

        if (summary.empty())
        {
            oss << "(no samples)";
            return oss.str();
        }

        oss << std::fixed << std::setprecision(3);
        for (const auto &stats : summary)
        {
            oss << "- " << stats.name
                << ": calls=" << stats.calls
                << ", total_ms=" << stats.total_ms
                << ", avg_ms=" << stats.average_ms()
                << ", min_ms=" << stats.min_ms
                << ", max_ms=" << stats.max_ms << "\n";
        }
        return oss.str();
    }

    std::string format_profile_csv()
    {
        const auto summary = profile_summary();
        std::ostringstream oss;
        oss << "name,calls,total_ms,avg_ms,min_ms,max_ms\n";
        oss << std::fixed << std::setprecision(3);

        for (const auto &stats : summary)
        {
            oss << escape_csv(stats.name) << ","
                << stats.calls << ","
                << stats.total_ms << ","
                << stats.average_ms() << ","
                << stats.min_ms << ","
                << stats.max_ms << "\n";
        }

        return oss.str();
    }

    std::string format_profile_json()
    {
        const auto summary = profile_summary();
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3);
        oss << "[";

        for (std::size_t i = 0; i < summary.size(); ++i)
        {
            if (i > 0)
            {
                oss << ",";
            }

            const auto &stats = summary[i];
            oss << "{"
                << "\"name\":\"" << escape_json(stats.name) << "\""
                << ",\"calls\":" << stats.calls
                << ",\"total_ms\":" << stats.total_ms
                << ",\"avg_ms\":" << stats.average_ms()
                << ",\"min_ms\":" << stats.min_ms
                << ",\"max_ms\":" << stats.max_ms
                << "}";
        }

        oss << "]";
        return oss.str();
    }

    ScopedProfile::ScopedProfile(std::string name, bool active) noexcept
        : name_(std::move(name)),
          active_(active && profiling_enabled()),
          stopped_(false),
          start_(clock::now()),
          recorded_ms_(0.0)
    {
    }

    ScopedProfile::~ScopedProfile()
    {
        stop();
    }

    void ScopedProfile::stop() noexcept
    {
        if (!active_ || stopped_)
        {
            return;
        }

        recorded_ms_ = elapsed_ms();
        stopped_ = true;
        record_profile_event(name_, recorded_ms_);
    }

    double ScopedProfile::elapsed_ms() const noexcept
    {
        if (stopped_)
        {
            return recorded_ms_;
        }

        const auto now = clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

} // namespace synara
