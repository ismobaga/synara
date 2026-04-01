#pragma once

#include "synara/nn/module.hpp"
#include "synara/ops/activation.hpp"

namespace synara
{

    class Sigmoid : public Module
    {
    public:
        Tensor forward(const Tensor &input) override { return sigmoid(input); }
        std::vector<Parameter *> parameters() override { return {}; }
    };

} // namespace synara
