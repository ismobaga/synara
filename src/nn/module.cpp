#include "synara/nn/module.hpp"

namespace synara
{

    std::vector<Parameter *> Module::parameters()
    {
        return {};
    }

    StateDict Module::state_dict(const std::string &) const
    {
        return {};
    }

    void Module::load_state_dict(const StateDict &, const std::string &)
    {
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
