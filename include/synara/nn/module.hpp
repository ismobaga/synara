#pragma once

#include <vector>

#include "synara/nn/parameter.hpp"
#include "synara/tensor/tensor.hpp"

namespace synara
{

    class Module
    {
    public:
        virtual ~Module() = default;

        virtual Tensor forward(const Tensor &input) = 0;
        virtual std::vector<Parameter *> parameters();

        Tensor operator()(const Tensor &input);
        void zero_grad();
    };

} // namespace synara
