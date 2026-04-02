#pragma once

#include "synara/autograd/node.hpp"
#include "synara/tensor/tensor.hpp"

namespace synara
{

    class AddNode : public Node
    {
    public:
        AddNode(const Tensor &a, const Tensor &b);

        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
        Tensor b_;
    };

    class SubNode : public Node
    {
    public:
        SubNode(const Tensor &a, const Tensor &b);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
        Tensor b_;
    };
    class MulNode : public Node
    {
    public:
        MulNode(const Tensor &a, const Tensor &b);

        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
        Tensor b_;
    };

    class DivNode : public Node
    {
    public:
        DivNode(const Tensor &a, const Tensor &b);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
        Tensor b_;
    };

    class SumNode : public Node
    {
    public:
        explicit SumNode(const Tensor &a);

        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
    };

    class MeanNode : public Node
    {
    public:
        explicit MeanNode(const Tensor &a);

        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
    };

    class ReLUNode : public Node
    {
    public:
        explicit ReLUNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
    };

    class LeakyReLUNode : public Node
    {
    public:
        LeakyReLUNode(const Tensor &a, Tensor::value_type negative_slope);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
        Tensor::value_type negative_slope_;
    };

    class SigmoidNode : public Node
    {
    public:
        explicit SigmoidNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
    };

    class TanhNode : public Node
    {
    public:
        explicit TanhNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
    };

    class SoftmaxNode : public Node
    {
    public:
        SoftmaxNode(const Tensor &a, int dim, const Tensor &output);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
        int dim_;
        Tensor output_;
    };

    class MatMulNode : public Node
    {
    public:
        MatMulNode(const Tensor &a, const Tensor &b);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
        Tensor b_;
    };

    class GELUNode : public Node
    {
    public:
        explicit GELUNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor a_;
    };

    class CrossEntropyNode : public Node
    {
    public:
        CrossEntropyNode(const Tensor &logits, const Tensor &targets);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor logits_;
        Tensor targets_;
    };

    class BCENode : public Node
    {
    public:
        BCENode(const Tensor &pred, const Tensor &target, Tensor::value_type eps);
        void backward(const Tensor &grad_output) override;

    private:
        Tensor pred_;
        Tensor target_;
        Tensor::value_type eps_;
    };
} // namespace synara