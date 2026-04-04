#include <chrono>
#include <iostream>
#include <string>

#include "synara.hpp"

int main(int argc, char **argv)
{
    using namespace synara;

    Tensor::manual_seed(42);

    std::cout << "OpenMP enabled: " << (openmp_enabled() ? "yes" : "no") << "\n";
    std::cout << "Threads before tuning: " << get_num_threads() << "\n";

    apply_autotuned_parallel_config(1 << 18, 4);
    const ParallelConfig &cfg = parallel_config();

    std::cout << "Autotuned thresholds:\n";
    std::cout << "  elementwise: " << cfg.elementwise_threshold << "\n";
    std::cout << "  matmul:      " << cfg.matmul_threshold << "\n";
    std::cout << "  linear:      " << cfg.linear_threshold << "\n";
    std::cout << "  conv2d:      " << cfg.conv2d_threshold << "\n";
    std::cout << "  pooling:     " << cfg.pooling_threshold << "\n";
    std::cout << "Threads after tuning: " << get_num_threads() << "\n\n";

    auto a = Tensor::uniform(Shape({256, 256}), -1.0, 1.0);
    auto b = Tensor::uniform(Shape({256, 256}), -1.0, 1.0);

    reset_profile_data();

    auto t0 = std::chrono::steady_clock::now();
    Tensor c;
    {
        ScopedProfile profile("matmul_demo");
        c = matmul(a, b);
    }
    auto t1 = std::chrono::steady_clock::now();

    auto x = Tensor::uniform(Shape({4, 8, 32, 32}), -1.0, 1.0);
    auto w = Tensor::uniform(Shape({16, 8, 3, 3}), -1.0, 1.0);
    auto bias = Tensor::uniform(Shape({16}), -0.1, 0.1);

    auto t2 = std::chrono::steady_clock::now();
    Tensor y;
    {
        ScopedProfile profile("conv2d_demo");
        y = conv2d(x, w, bias, 1, 1, 1, 1);
    }
    auto t3 = std::chrono::steady_clock::now();

    Tensor p;
    {
        ScopedProfile profile("maxpool_demo");
        p = max_pool2d(y, 2, 2, 2, 2, 0, 0);
    }
    auto t4 = std::chrono::steady_clock::now();

    const auto matmul_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const auto conv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    const auto pool_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();

    std::cout << "matmul_ms=" << matmul_ms << "\n";
    std::cout << "conv2d_ms=" << conv_ms << "\n";
    std::cout << "maxpool_ms=" << pool_ms << "\n";
    std::cout << "sample=" << c.at({0, 0}) + p.at({0, 0, 0, 0}) << "\n\n";

    std::cout << format_profile_summary();

    if (argc > 1)
    {
        const std::string prefix = argv[1];
        const std::string csv_path = prefix + ".csv";
        const std::string json_path = prefix + ".json";

        const bool csv_ok = write_profile_csv(csv_path);
        const bool json_ok = write_profile_json(json_path);
        std::cout << "\nSaved profile CSV: " << (csv_ok ? csv_path : "<failed>") << "\n";
        std::cout << "Saved profile JSON: " << (json_ok ? json_path : "<failed>") << "\n";
    }

    return 0;
}
