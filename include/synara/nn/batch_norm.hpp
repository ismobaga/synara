#pragma once

#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"

namespace synara
{

    class BatchNorm1d : public Module
    {
    public:
        BatchNorm1d(Size num_features,
                    bool affine = true,
                    Tensor::value_type eps = 1e-5f,
                    Tensor::value_type momentum = 0.1f);

        Tensor forward(const Tensor &input) override;
        std::vector<Parameter *> parameters() override;
        StateDict state_dict(const std::string &prefix = "") const override;
        void load_state_dict(const StateDict &state, const std::string &prefix = "") override;

        void train() noexcept;
        void eval() noexcept;
        bool is_training() const noexcept;

        Size num_features() const noexcept;
        bool affine() const noexcept;

        Parameter &weight() noexcept;
        const Parameter &weight() const noexcept;

        Parameter &bias() noexcept;
        const Parameter &bias() const noexcept;

        const Tensor &running_mean() const noexcept;
        const Tensor &running_var() const noexcept;

    private:
        Size num_features_;
        bool affine_;
        Tensor::value_type eps_;
        Tensor::value_type momentum_;
        bool training_;

        Parameter gamma_;
        Parameter beta_;
        Tensor running_mean_;
        Tensor running_var_;
    };

    class BatchNorm2d : public Module
    {
    public:
        BatchNorm2d(Size num_features,
                    bool affine = true,
                    Tensor::value_type eps = 1e-5f,
                    Tensor::value_type momentum = 0.1f);

        Tensor forward(const Tensor &input) override;
        std::vector<Parameter *> parameters() override;
        StateDict state_dict(const std::string &prefix = "") const override;
        void load_state_dict(const StateDict &state, const std::string &prefix = "") override;

        void train() noexcept;
        void eval() noexcept;
        bool is_training() const noexcept;

        Size num_features() const noexcept;
        bool affine() const noexcept;

        Parameter &weight() noexcept;
        const Parameter &weight() const noexcept;

        Parameter &bias() noexcept;
        const Parameter &bias() const noexcept;

        const Tensor &running_mean() const noexcept;
        const Tensor &running_var() const noexcept;

    private:
        Size num_features_;
        bool affine_;
        Tensor::value_type eps_;
        Tensor::value_type momentum_;
        bool training_;

        Parameter gamma_;
        Parameter beta_;
        Tensor running_mean_;
        Tensor running_var_;
    };

} // namespace synara
