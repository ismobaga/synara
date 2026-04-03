#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include "synara/nn/module.hpp"

namespace synara
{

    // Parameter counting utilities
    struct ParameterInfo
    {
        std::string name;
        std::string module_type;
        std::vector<std::size_t> shape;
        std::size_t numel = 0;
        bool requires_grad = false;

        ParameterInfo() = default;
        ParameterInfo(const std::string &name, const std::string &module_type,
                      const std::vector<std::size_t> &shape, std::size_t numel, bool requires_grad)
            : name(name), module_type(module_type), shape(shape), numel(numel), requires_grad(requires_grad)
        {
        }
    };

    struct ModuleStatistics
    {
        std::string module_type;
        std::size_t total_parameters = 0;
        std::size_t trainable_parameters = 0;
        std::size_t non_trainable_parameters = 0;
    };

    // Count total parameters in a module (including all submodules)
    std::size_t total_parameters(const Module &module);

    // Count trainable parameters (requires_grad == true)
    std::size_t trainable_parameters(const Module &module);

    // Count non-trainable parameters (requires_grad == false)
    std::size_t non_trainable_parameters(const Module &module);

    // Get detailed information about all parameters
    std::vector<ParameterInfo> parameter_info(const Module &module);

    // Get statistics per module type
    std::unordered_map<std::string, ModuleStatistics> module_statistics(const Module &module);

    // Pretty-print parameter summary
    std::string parameter_summary(const Module &module);

    // Get memory usage in bytes (assumes double precision = 8 bytes per value)
    std::size_t memory_usage_bytes(const Module &module);

    // Pretty-print memory summary
    std::string memory_summary(const Module &module);

} // namespace synara
