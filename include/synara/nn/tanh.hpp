#pragma once

#include "synara/nn/module.hpp"
#include "synara/ops/activation.hpp"

namespace synara
{

    class Tanh : public Module
    {
    public:
        Tensor forward(const Tensor &input) override { return tanh(input); }
        std::vector<Parameter *> parameters() override { return {}; }
    };

} // namespace synara
