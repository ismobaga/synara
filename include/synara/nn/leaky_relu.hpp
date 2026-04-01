#pragma once

#include "synara/nn/module.hpp"
#include "synara/ops/activation.hpp"

namespace synara
{

    class LeakyReLU : public Module
    {
    public:
        explicit LeakyReLU(Tensor::value_type negative_slope = 0.01f)
            : negative_slope_(negative_slope) {}

        Tensor forward(const Tensor &input) override { return leaky_relu(input, negative_slope_); }
        std::vector<Parameter *> parameters() override { return {}; }

    private:
        Tensor::value_type negative_slope_;
    };

} // namespace synara
