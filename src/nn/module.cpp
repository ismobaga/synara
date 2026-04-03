#include "synara/nn/module.hpp"

#include <typeinfo>
#include <sstream>
#include <iomanip>

namespace synara
{

    std::vector<Parameter *> Module::parameters()
    {
        return {};
    }

    std::vector<std::pair<std::string, Tensor *>> Module::named_parameters(const std::string &prefix)
    {
        std::vector<std::pair<std::string, Tensor *>> out;
        std::vector<Parameter *> params = parameters();
        out.reserve(params.size());
        for (Size i = 0; i < params.size(); ++i)
        {
            out.emplace_back(prefix + "param_" + std::to_string(i), &params[i]->tensor());
        }
        return out;
    }

    std::vector<std::pair<std::string, Module *>> Module::named_modules(const std::string &prefix)
    {
        return {{prefix, this}};
    }

    StateDict Module::state_dict(const std::string &) const
    {
        return {};
    }

    void Module::load_state_dict(const StateDict &, const std::string &)
    {
    }

    void Module::train() noexcept
    {
        training_ = true;
    }

    void Module::eval() noexcept
    {
        training_ = false;
    }

    bool Module::is_training() const noexcept
    {
        return training_;
    }

    std::string Module::to_string() const
    {
        return typeid(*this).name();
    }

    Tensor Module::operator()(const Tensor &input)
    {
        return forward(input);
    }

    void Module::zero_grad()
    {
        for (Parameter *parameter : parameters())
        {
            parameter->tensor().zero_grad();
        }
    }

    // Enhanced introspection implementations

    std::size_t Module::num_parameters() const
    {
        std::size_t count = 0;
        for (const Parameter *param : const_cast<Module *>(this)->parameters())
        {
            count += param->tensor().numel();
        }
        return count;
    }

    std::size_t Module::num_trainable_parameters() const
    {
        std::size_t count = 0;
        for (const Parameter *param : const_cast<Module *>(this)->parameters())
        {
            if (param->tensor().requires_grad())
            {
                count += param->tensor().numel();
            }
        }
        return count;
    }

    std::string Module::parameter_tree(const std::string &prefix) const
    {
        std::ostringstream oss;
        auto named_params = const_cast<Module *>(this)->named_parameters(prefix);

        for (const auto &[name, tensor_ptr] : named_params)
        {
            if (tensor_ptr)
            {
                oss << name << ": shape [";
                for (std::size_t d = 0; d < tensor_ptr->rank(); ++d)
                {
                    if (d > 0)
                        oss << ", ";
                    oss << tensor_ptr->shape()[d];
                }
                oss << "] (size: " << tensor_ptr->numel() << ", grad: " << (tensor_ptr->requires_grad() ? "yes" : "no") << ")\n";
            }
        }
        return oss.str();
    }

    std::string Module::module_tree(const std::string &prefix) const
    {
        std::ostringstream oss;
        auto named_mods = const_cast<Module *>(this)->named_modules(prefix);

        for (const auto &[name, mod_ptr] : named_mods)
        {
            if (mod_ptr)
            {
                std::string mod_name = mod_ptr->to_string();
                // Try to demangle the type name if possible
                size_t pos = mod_name.rfind(':');
                if (pos != std::string::npos)
                {
                    mod_name = mod_name.substr(pos + 1);
                }

                oss << (name.empty() ? "root" : name) << ": " << mod_name << "\n";
                std::size_t params = mod_ptr->num_parameters();
                if (params > 0)
                {
                    oss << "  ├─ parameters: " << params << " (trainable: " << mod_ptr->num_trainable_parameters() << ")\n";
                }
            }
        }
        return oss.str();
    }

    std::vector<std::pair<std::string, std::vector<std::size_t>>> Module::parameter_shapes() const
    {
        std::vector<std::pair<std::string, std::vector<std::size_t>>> result;
        auto named_params = const_cast<Module *>(this)->named_parameters();

        for (const auto &[name, tensor_ptr] : named_params)
        {
            if (tensor_ptr)
            {
                std::vector<std::size_t> shape;
                for (std::size_t d = 0; d < tensor_ptr->rank(); ++d)
                {
                    shape.push_back(tensor_ptr->shape()[d]);
                }
                result.emplace_back(name, shape);
            }
        }
        return result;
    }

    std::size_t Module::memory_usage() const
    {
        // Assumes double precision (8 bytes per value)
        return num_parameters() * sizeof(double);
    }

} // namespace synara
