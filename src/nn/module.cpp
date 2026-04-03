#include "synara/nn/module.hpp"

#include <typeinfo>

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

} // namespace synara
