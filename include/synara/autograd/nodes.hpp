#pragma once
#include "synara/autograd/node.hpp"
#include "synara/tensor/tensor.hpp"
#include <vector>
namespace synara
{
    class AddNode : public Node
    {
    public:
        AddNode(const Tensor &a, const Tensor &b);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_, &b_}; }

    private:
        Tensor a_;
        Tensor b_;
    };
    class SubNode : public Node
    {
    public:
        SubNode(const Tensor &a, const Tensor &b);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_, &b_}; }

    private:
        Tensor a_;
        Tensor b_;
    };
    class MulNode : public Node
    {
    public:
        MulNode(const Tensor &a, const Tensor &b);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_, &b_}; }

    private:
        Tensor a_;
        Tensor b_;
    };
    class DivNode : public Node
    {
    public:
        DivNode(const Tensor &a, const Tensor &b);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_, &b_}; }

    private:
        Tensor a_;
        Tensor b_;
    };
    class SumNode : public Node
    {
    public:
        explicit SumNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class MeanNode : public Node
    {
    public:
        explicit MeanNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class ReLUNode : public Node
    {
    public:
        explicit ReLUNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class LeakyReLUNode : public Node
    {
    public:
        LeakyReLUNode(const Tensor &a, Tensor::value_type negative_slope);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        Tensor::value_type negative_slope_;
    };
    class SigmoidNode : public Node
    {
    public:
        explicit SigmoidNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class TanhNode : public Node
    {
    public:
        explicit TanhNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class SoftmaxNode : public Node
    {
    public:
        SoftmaxNode(const Tensor &a, int dim, const Tensor &output);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

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
        std::vector<Tensor *> inputs() override { return {&a_, &b_}; }

    private:
        Tensor a_;
        Tensor b_;
    };
    class GELUNode : public Node
    {
    public:
        explicit GELUNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class CrossEntropyNode : public Node
    {
    public:
        CrossEntropyNode(const Tensor &logits, const Tensor &targets);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&logits_}; }

    private:
        Tensor logits_;
        Tensor targets_;
    };
    class BCENode : public Node
    {
    public:
        BCENode(const Tensor &pred, const Tensor &target, Tensor::value_type eps);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&pred_}; }

    private:
        Tensor pred_;
        Tensor target_;
        Tensor::value_type eps_;
    };
    // ---- Dim-aware Reduction Nodes ----
    class SumDimNode : public Node
    {
    public:
        SumDimNode(const Tensor &a, int dim, bool keepdim);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        int dim_;
        bool keepdim_;
    };
    class MeanDimNode : public Node
    {
    public:
        MeanDimNode(const Tensor &a, int dim, bool keepdim);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        int dim_;
        bool keepdim_;
    };
    class MaxDimNode : public Node
    {
    public:
        MaxDimNode(const Tensor &a, int dim, bool keepdim, const Tensor &output);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        int dim_;
        bool keepdim_;
        Tensor output_;
    };
    class MinDimNode : public Node
    {
    public:
        MinDimNode(const Tensor &a, int dim, bool keepdim, const Tensor &output);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        int dim_;
        bool keepdim_;
        Tensor output_;
    };
    // ---- Shape Nodes ----
    class SqueezeNode : public Node
    {
    public:
        SqueezeNode(const Tensor &a, int dim);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        int dim_;
    };
    class UnsqueezeNode : public Node
    {
    public:
        UnsqueezeNode(const Tensor &a, int dim);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        int dim_;
    };
    class PermuteNode : public Node
    {
    public:
        PermuteNode(const Tensor &a, const std::vector<int> &dims);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        std::vector<int> dims_;
    };
    class CatNode : public Node
    {
    public:
        CatNode(const std::vector<Tensor> &inputs, int dim);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override
        {
            std::vector<Tensor *> ptrs;
            ptrs.reserve(inputs_.size());
            for (auto &t : inputs_)
                ptrs.push_back(&t);
            return ptrs;
        }

    private:
        std::vector<Tensor> inputs_;
        int dim_;
    };
    class StackNode : public Node
    {
    public:
        StackNode(const std::vector<Tensor> &inputs, int dim);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override
        {
            std::vector<Tensor *> ptrs;
            ptrs.reserve(inputs_.size());
            for (auto &t : inputs_)
                ptrs.push_back(&t);
            return ptrs;
        }

    private:
        std::vector<Tensor> inputs_;
        int dim_;
    };
    class SplitNode : public Node
    {
    public:
        SplitNode(const Tensor &a, int split_size, int dim);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }
        void register_output(std::size_t idx, Tensor &out);

    private:
        Tensor a_;
        int split_size_;
        int dim_;
        std::vector<std::weak_ptr<void>> outputs_;
    };
    class SplitPieceNode : public Node
    {
    public:
        SplitPieceNode(const Tensor &a, int dim, Size offset, Size chunk);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        int dim_;
        Size offset_;
        Size chunk_;
    };
    // ---- Math Nodes ----
    class ExpNode : public Node
    {
    public:
        ExpNode(const Tensor &a, const Tensor &output);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        Tensor output_;
    };
    class LogNode : public Node
    {
    public:
        explicit LogNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class Log2Node : public Node
    {
    public:
        explicit Log2Node(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class SqrtNode : public Node
    {
    public:
        SqrtNode(const Tensor &a, const Tensor &output);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        Tensor output_;
    };
    class PowNode : public Node
    {
    public:
        PowNode(const Tensor &a, Tensor::value_type exponent);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        Tensor::value_type exponent_;
    };
    class AbsNode : public Node
    {
    public:
        explicit AbsNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class SignNode : public Node
    {
    public:
        explicit SignNode(const Tensor &a);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
    };
    class ClampNode : public Node
    {
    public:
        ClampNode(const Tensor &a, Tensor::value_type min_val, Tensor::value_type max_val);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&a_}; }

    private:
        Tensor a_;
        Tensor::value_type min_val_;
        Tensor::value_type max_val_;
    };
    class EmbeddingNode : public Node
    {
    public:
        EmbeddingNode(Tensor indices, Tensor weight);
        void backward(const Tensor &grad_output) override;
        std::vector<Tensor *> inputs() override { return {&weight_}; }

    private:
        Tensor indices_;
        Tensor weight_;
    };
} // namespace synara