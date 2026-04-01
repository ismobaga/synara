#pragma once

#include "synara/nn/module.hpp"
#include "synara/ops/activation.hpp"

namespace synara
{

    class Softmax : public Module
    {
    public:
        explicit Softmax(int dim = -1) : dim_(dim) {}

        Tensor forward(const Tensor &input) override { return softmax(input, dim_); }
        std::vector<Parameter *> parameters() override { return {}; }

    private:
        int dim_;
    };

} // namespace synara
