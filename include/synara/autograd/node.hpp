#pragma once

#include <memory>
#include <vector>

namespace synara {

class Tensor;

class Node {
public:
    virtual ~Node() = default;
    virtual void backward(const Tensor& grad_output) = 0;
};

} // namespace synara