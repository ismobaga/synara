#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
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

    const std::string csv = format_profile_csv();
    assert(csv.find("name,calls,total_ms,avg_ms,min_ms,max_ms") != std::string::npos);
    assert(csv.find("sleep_scope") != std::string::npos);

    const std::string json = format_profile_json();
    assert(json.find("\"name\":\"sleep_scope\"") != std::string::npos);
    assert(json.find("\"calls\":1") != std::string::npos);

    const std::filesystem::path csv_path = "synara_profile_test.csv";
    const std::filesystem::path json_path = "synara_profile_test.json";
    assert(write_profile_csv(csv_path.string()));
    assert(write_profile_json(json_path.string()));

    std::ifstream csv_in(csv_path);
    std::stringstream csv_buffer;
    csv_buffer << csv_in.rdbuf();
    assert(csv_buffer.str().find("sleep_scope") != std::string::npos);

    std::ifstream json_in(json_path);
    std::stringstream json_buffer;
    json_buffer << json_in.rdbuf();
    assert(json_buffer.str().find("\"sleep_scope\"") != std::string::npos);

    std::filesystem::remove(csv_path);
    std::filesystem::remove(json_path);

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
