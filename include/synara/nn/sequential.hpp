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

        Size size() const noexcept;

    private:
        std::vector<std::shared_ptr<Module>> modules_;
    };

} // namespace synara
