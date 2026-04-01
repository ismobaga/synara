#pragma once

#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"

namespace synara
{

    class Conv2d : public Module
    {
    public:
        Conv2d(Size in_channels,
               Size out_channels,
               Size kernel_h,
               Size kernel_w,
               Size stride_h = 1,
               Size stride_w = 1,
               Size pad_h = 0,
               Size pad_w = 0,
             bool use_bias = true,
             Size dilation_h = 1,
             Size dilation_w = 1,
             Size groups = 1);

        Tensor forward(const Tensor &input) override;
        std::vector<Parameter *> parameters() override;
        StateDict state_dict(const std::string &prefix = "") const override;
        void load_state_dict(const StateDict &state, const std::string &prefix = "") override;

        Parameter &weight() noexcept;
        const Parameter &weight() const noexcept;

        Parameter &bias() noexcept;
        const Parameter &bias() const noexcept;

        Size in_channels() const noexcept;
        Size out_channels() const noexcept;
        Size kernel_h() const noexcept;
        Size kernel_w() const noexcept;
        Size stride_h() const noexcept;
        Size stride_w() const noexcept;
        Size pad_h() const noexcept;
        Size pad_w() const noexcept;
        Size dilation_h() const noexcept;
        Size dilation_w() const noexcept;
        Size groups() const noexcept;
        bool has_bias() const noexcept;

    private:
        Parameter make_weight() const;
        Parameter make_bias() const;

        Size in_channels_;
        Size out_channels_;
        Size kernel_h_;
        Size kernel_w_;
        Size stride_h_;
        Size stride_w_;
        Size pad_h_;
        Size pad_w_;
        Size dilation_h_;
        Size dilation_w_;
        Size groups_;
        bool use_bias_;

        Parameter weight_;
        Parameter bias_;
    };

} // namespace synara
