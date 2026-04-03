#pragma once

#include <string>
#include <utility>
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
        Module() = default;
        virtual ~Module() = default;

        virtual Tensor forward(const Tensor &input) = 0;
        virtual std::vector<Parameter *> parameters();
        virtual std::vector<std::pair<std::string, Tensor *>> named_parameters(const std::string &prefix = "");
        virtual std::vector<std::pair<std::string, Module *>> named_modules(const std::string &prefix = "");
        virtual StateDict state_dict(const std::string &prefix = "") const;
        virtual void load_state_dict(const StateDict &state, const std::string &prefix = "");
        virtual void train() noexcept;
        virtual void eval() noexcept;
        virtual bool is_training() const noexcept;
        virtual std::string to_string() const;

        Tensor operator()(const Tensor &input);
        void zero_grad();

    protected:
        bool training_ = true;
    };

} // namespace synara
