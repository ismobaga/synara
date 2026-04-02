#pragma once

namespace synara
{

class Tensor;

/// Accumulates gradients through the entire computation graph reachable from
/// @p output by invoking each Node's backward() method in reverse topological
/// order.
///
/// All leaf tensors (requires_grad == true, is_leaf == true) that contributed
/// to @p output will have their grad() populated after this call.
///
/// Equivalent to calling output.backward(), provided as a free function for
/// convenience.
///
/// @param output  A scalar tensor (numel() == 1) representing the loss or any
///                differentiable quantity to differentiate.
void accumulate_gradients(Tensor &output);

} // namespace synara
