#pragma once

#include <cstddef>
#include <vector>

#include "synara/core/types.hpp"
#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"

namespace synara
{

    class Linear : public Module
    {
    public:
        Linear(Size in_features, Size out_features, bool use_bias = true);

        Tensor forward(const Tensor &input) override;
        std::vector<Parameter *> parameters() override;
        StateDict state_dict(const std::string &prefix = "") const override;
        void load_state_dict(const StateDict &state, const std::string &prefix = "") override;

        Parameter &weight() noexcept;
        const Parameter &weight() const noexcept;

        Parameter &bias() noexcept;
        const Parameter &bias() const noexcept;

        Size in_features() const noexcept;
        Size out_features() const noexcept;
        bool has_bias() const noexcept;

    private:
        Parameter make_weight(Size in_features, Size out_features) const;
        Parameter make_bias(Size out_features) const;

        Size in_features_;
        Size out_features_;
        bool use_bias_;
        Parameter weight_;
        Parameter bias_;
    };

} // namespace synara
