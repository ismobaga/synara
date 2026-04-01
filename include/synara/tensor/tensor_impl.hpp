#pragma once

#include <memory>
#include <vector>

#include "synara/core/types.hpp"
#include "synara/tensor/storage.hpp"
#include "synara/tensor/shape.hpp"
#include "synara/tensor/strides.hpp"

namespace synara {

class Tensor;
class Node;

struct TensorImpl {
    Shape shape;
    Strides strides;
    std::shared_ptr<Storage> storage;
    Size offset = 0;

    // Autograd metadata
    bool requires_grad = false;
    bool is_leaf = true;
    std::shared_ptr<Tensor> grad;
    std::shared_ptr<Node> grad_fn; 

    TensorImpl()
        : shape(),
          strides(Strides::contiguous(shape)),
          storage(std::make_shared<Storage>(1)),
          offset(0),
          requires_grad(false),
          is_leaf(true),
          grad(nullptr),
          grad_fn(nullptr) {}

    TensorImpl(
        Shape shape_,
        Strides strides_,
        std::shared_ptr<Storage> storage_,
        Size offset_,
        bool requires_grad_,
        bool is_leaf_)
        : shape(std::move(shape_)),
          strides(std::move(strides_)),
          storage(std::move(storage_)),
          offset(offset_),
          requires_grad(requires_grad_),
          is_leaf(is_leaf_),
          grad(nullptr),
          grad_fn(nullptr) {}
};

} // namespace synara