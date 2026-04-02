#pragma once

#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"

namespace synara
{

    class LayerNorm : public Module
    {
    public:
        LayerNorm(Size num_features,
                  bool affine = true,
                  Tensor::value_type eps = 1e-5f);

        Tensor forward(const Tensor &input) override;
        std::vector<Parameter *> parameters() override;
        StateDict state_dict(const std::string &prefix = "") const override;
        void load_state_dict(const StateDict &state, const std::string &prefix = "") override;

        Size num_features() const noexcept;
        bool affine() const noexcept;

        Parameter &weight() noexcept;
        const Parameter &weight() const noexcept;

        Parameter &bias() noexcept;
        const Parameter &bias() const noexcept;

    private:
        Size num_features_;
        bool affine_;
        Tensor::value_type eps_;
        Parameter gamma_;
        Parameter beta_;
    };

} // namespace synara