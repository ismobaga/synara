#include "synara/nn/sequential.hpp"

#include <string>
#include <utility>

namespace synara
{

    Sequential::Sequential(std::vector<std::shared_ptr<Module>> modules)
        : modules_(std::move(modules)) {}

    void Sequential::add(std::shared_ptr<Module> module)
    {
        modules_.push_back(std::move(module));
    }

    Tensor Sequential::forward(const Tensor &input)
    {
        Tensor output = input;
        for (const std::shared_ptr<Module> &module : modules_)
        {
            output = (*module)(output);
        }
        return output;
    }

    std::vector<Parameter *> Sequential::parameters()
    {
        std::vector<Parameter *> out;
        for (const std::shared_ptr<Module> &module : modules_)
        {
            std::vector<Parameter *> child_params = module->parameters();
            out.insert(out.end(), child_params.begin(), child_params.end());
        }
        return out;
    }

    StateDict Sequential::state_dict(const std::string &prefix) const
    {
        StateDict out;
        for (Size i = 0; i < modules_.size(); ++i)
        {
            const std::string module_prefix = prefix + "layers." + std::to_string(i) + ".";
            StateDict child = modules_[i]->state_dict(module_prefix);
            out.insert(child.begin(), child.end());
        }
        return out;
    }

    void Sequential::load_state_dict(const StateDict &state, const std::string &prefix)
    {
        for (Size i = 0; i < modules_.size(); ++i)
        {
            const std::string module_prefix = prefix + "layers." + std::to_string(i) + ".";
            modules_[i]->load_state_dict(state, module_prefix);
        }
    }

    Size Sequential::size() const noexcept
    {
        return modules_.size();
    }

} // namespace synara
