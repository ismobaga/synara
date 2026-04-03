#include "synara/nn/module_utils.hpp"

#include <iomanip>
#include <sstream>
#include <cmath>

namespace synara
{

    std::size_t total_parameters(const Module &module)
    {
        auto params = const_cast<Module &>(module).parameters();
        std::size_t total = 0;
        for (const auto *param : params)
        {
            total += param->tensor().numel();
        }
        return total;
    }

    std::size_t trainable_parameters(const Module &module)
    {
        auto params = const_cast<Module &>(module).parameters();
        std::size_t trainable = 0;
        for (const auto *param : params)
        {
            if (param->tensor().requires_grad())
            {
                trainable += param->tensor().numel();
            }
        }
        return trainable;
    }

    std::size_t non_trainable_parameters(const Module &module)
    {
        auto params = const_cast<Module &>(module).parameters();
        std::size_t non_trainable = 0;
        for (const auto *param : params)
        {
            if (!param->tensor().requires_grad())
            {
                non_trainable += param->tensor().numel();
            }
        }
        return non_trainable;
    }

    std::vector<ParameterInfo> parameter_info(const Module &module)
    {
        std::vector<ParameterInfo> result;
        auto named_params = const_cast<Module &>(module).named_parameters();

        for (const auto &[name, tensor_ptr] : named_params)
        {
            if (tensor_ptr)
            {
                const Tensor &t = *tensor_ptr;
                std::vector<std::size_t> shape;
                for (std::size_t d = 0; d < t.rank(); ++d)
                {
                    shape.push_back(t.shape()[d]);
                }

                result.emplace_back(name, "Parameter", shape, t.numel(), t.requires_grad());
            }
        }

        return result;
    }

    std::unordered_map<std::string, ModuleStatistics> module_statistics(const Module &module)
    {
        std::unordered_map<std::string, ModuleStatistics> stats;

        auto named_modules = const_cast<Module &>(module).named_modules();
        for (const auto &[name, mod_ptr] : named_modules)
        {
            if (mod_ptr)
            {
                std::string module_type = mod_ptr->to_string();
                if (stats.find(module_type) == stats.end())
                {
                    stats[module_type] = ModuleStatistics{module_type, 0, 0, 0};
                }

                auto params = mod_ptr->parameters();
                std::size_t total = 0;
                std::size_t trainable = 0;
                std::size_t non_trainable = 0;

                for (const auto *param : params)
                {
                    std::size_t numel = param->tensor().numel();
                    total += numel;
                    if (param->tensor().requires_grad())
                    {
                        trainable += numel;
                    }
                    else
                    {
                        non_trainable += numel;
                    }
                }

                stats[module_type].total_parameters += total;
                stats[module_type].trainable_parameters += trainable;
                stats[module_type].non_trainable_parameters += non_trainable;
            }
        }

        return stats;
    }

    static std::string format_number(std::size_t num)
    {
        if (num >= 1000000)
        {
            return std::to_string(num / 1000000) + "M";
        }
        else if (num >= 1000)
        {
            return std::to_string(num / 1000) + "K";
        }
        else
        {
            return std::to_string(num);
        }
    }

    static std::string format_bytes(std::size_t bytes)
    {
        if (bytes >= 1024 * 1024 * 1024)
        {
            double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << gb << " GB";
            return oss.str();
        }
        else if (bytes >= 1024 * 1024)
        {
            double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mb << " MB";
            return oss.str();
        }
        else if (bytes >= 1024)
        {
            double kb = static_cast<double>(bytes) / 1024.0;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << kb << " KB";
            return oss.str();
        }
        else
        {
            return std::to_string(bytes) + " B";
        }
    }

    std::string parameter_summary(const Module &module)
    {
        std::ostringstream oss;
        std::size_t total = total_parameters(module);
        std::size_t trainable = trainable_parameters(module);
        std::size_t non_trainable = non_trainable_parameters(module);

        oss << "=== Parameter Summary ===\n";
        oss << "Total Parameters:        " << std::setw(12) << format_number(total) << "\n";
        oss << "Trainable Parameters:    " << std::setw(12) << format_number(trainable) << "\n";
        oss << "Non-trainable Parameters:" << std::setw(12) << format_number(non_trainable) << "\n";

        if (total > 0)
        {
            double train_pct = (static_cast<double>(trainable) / static_cast<double>(total)) * 100.0;
            oss << "Trainable Percentage:    " << std::fixed << std::setprecision(1) << train_pct << "%\n";
        }

        return oss.str();
    }

    std::size_t memory_usage_bytes(const Module &module)
    {
        // Assumes double precision (8 bytes per value)
        return total_parameters(module) * sizeof(double);
    }

    std::string memory_summary(const Module &module)
    {
        std::ostringstream oss;
        std::size_t total_params = total_parameters(module);
        std::size_t total_bytes = memory_usage_bytes(module);

        oss << "=== Memory Summary ===\n";
        oss << "Total Parameters: " << format_number(total_params) << "\n";
        oss << "Memory Usage:     " << format_bytes(total_bytes) << "\n";
        oss << "Data Type:        double (8 bytes per value)\n";

        return oss.str();
    }

} // namespace synara
