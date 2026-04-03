#include "synara/autograd/engine.hpp"
#include "synara/autograd/backward.hpp"
#include "synara/tensor/tensor.hpp"

namespace synara
{

    // run_backward and accumulate_gradients are free-function aliases for
    // Tensor::backward().  They exist so callers can trigger differentiation
    // without holding an lvalue reference to the tensor method and to match the
    // declarations in engine.hpp / backward.hpp that document the autograd API.

    void run_backward(Tensor &root, bool retain_graph)
    {
        root.backward(retain_graph);
    }

    void accumulate_gradients(Tensor &output, bool retain_graph)
    {
        output.backward(retain_graph);
    }

} // namespace synara
