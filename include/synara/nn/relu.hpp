#pragma once

#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"
#include "synara/optim/optimizer.hpp"
#include "synara/ops/activation.hpp"

namespace synara
{

    class ReLU : public Module
    {
    public:
        Tensor forward(const Tensor &input) override { return relu(input); }
        std::vector<Parameter *> parameters() override { return {}; }
    };

} // namespace synara