#pragma once

#include "synara/autograd/node.hpp"
#include "synara/tensor/tensor.hpp"

namespace synara {

class AddNode : public Node {
public:
    AddNode(const Tensor& a, const Tensor& b);

    void backward(const Tensor& grad_output) override;

private:
    Tensor a_;
    Tensor b_;
};

class SubNode : public Node {
public:
    SubNode(const Tensor& a, const Tensor& b);
    void backward(const Tensor& grad_output) override;
private:
    Tensor a_;
    Tensor b_;
};
class MulNode : public Node {
public:
    MulNode(const Tensor& a, const Tensor& b);

    void backward(const Tensor& grad_output) override;

private:
    Tensor a_;
    Tensor b_;
};

class DivNode : public Node {
public:
    DivNode(const Tensor& a, const Tensor& b);
    void backward(const Tensor& grad_output) override;
private:
    Tensor a_;
    Tensor b_;
};

class SumNode : public Node {
public:
    explicit SumNode(const Tensor& a);

    void backward(const Tensor& grad_output) override;

private:
    Tensor a_;
};

class MeanNode : public Node {
public:
    explicit MeanNode(const Tensor& a);

    void backward(const Tensor& grad_output) override;  
private:    Tensor a_;
};

class ReLUNode : public Node {
public:
    explicit ReLUNode(const Tensor& a);
    void backward(const Tensor& grad_output) override;
private:
    Tensor a_;
};

class MatMulNode : public Node {
public:
    MatMulNode(const Tensor& a, const Tensor& b);
    void backward(const Tensor& grad_output) override;
private:
    Tensor a_;
    Tensor b_;
};
} // namespace synara