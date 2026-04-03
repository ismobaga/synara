#pragma once

#include <memory>
#include <vector>

#include "synara/nn/module.hpp"

namespace synara
{

    class Sequential : public Module
    {
    public:
        Sequential() = default;
        explicit Sequential(std::vector<std::shared_ptr<Module>> modules);

        void add(std::shared_ptr<Module> module);

        Tensor forward(const Tensor &input) override;
        std::vector<Parameter *> parameters() override;
        std::vector<std::pair<std::string, Tensor *>> named_parameters(const std::string &prefix = "") override;
        std::vector<std::pair<std::string, Module *>> named_modules(const std::string &prefix = "") override;
        StateDict state_dict(const std::string &prefix = "") const override;
        void load_state_dict(const StateDict &state, const std::string &prefix = "") override;
        void train() noexcept override;
        void eval() noexcept override;

        Size size() const noexcept;

    private:
        std::vector<std::shared_ptr<Module>> modules_;
    };

} // namespace synara
