#pragma once

namespace synara
{

class Tensor;

/// Runs the reverse-mode automatic differentiation pass starting from @p root.
///
/// The engine traverses the computation graph in reverse topological order,
/// invoking each Node's backward() method and accumulating gradients into all
/// leaf tensors that have requires_grad set to true.
///
/// This is the function invoked by Tensor::backward().  Calling it directly is
/// equivalent and can be useful when the root tensor is not directly accessible
/// as an lvalue.
///
/// @param root  The scalar (or scalar-loss) tensor from which to start the
///              backward pass.  Must have numel() == 1.
void run_backward(Tensor &root);

} // namespace synara
