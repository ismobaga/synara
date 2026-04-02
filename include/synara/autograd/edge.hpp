#pragma once

#include <memory>
#include <cstddef>

namespace synara
{

class Node;

/// An Edge represents a directed connection in the autograd computation graph.
/// Every non-leaf Tensor holds an Edge (its grad_fn) that points back to the
/// function that produced it.  The input_nr field identifies which input of
/// that function this particular edge corresponds to, which is relevant when a
/// single function has multiple differentiable inputs.
struct Edge
{
    /// The function (Node) that produced the associated tensor.
    std::shared_ptr<Node> node;

    /// Zero-based index of the input within @p node that this edge belongs to.
    std::size_t input_nr{0};

    /// Returns true if this edge points to a valid function.
    bool is_valid() const noexcept { return node != nullptr; }

    explicit operator bool() const noexcept { return is_valid(); }
};

} // namespace synara
