#pragma once

#include "synara/nn/module.hpp"
#include "synara/ops/activation.hpp"

namespace synara
{

    class GELU : public Module
    {
    public:
        Tensor forward(const Tensor &input) override { return gelu(input); }
        std::vector<Parameter *> parameters() override { return {}; }
    };

} // namespace synara
