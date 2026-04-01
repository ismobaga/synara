#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "synara/nn/parameter.hpp"
#include "synara/tensor/tensor.hpp"

namespace synara
{

    using StateDict = std::unordered_map<std::string, Tensor>;

    class Module
    {
    public:
        virtual ~Module() = default;

        virtual Tensor forward(const Tensor &input) = 0;
        virtual std::vector<Parameter *> parameters();
        virtual StateDict state_dict(const std::string &prefix = "") const;
        virtual void load_state_dict(const StateDict &state, const std::string &prefix = "");

        Tensor operator()(const Tensor &input);
        void zero_grad();
    };

} // namespace synara
