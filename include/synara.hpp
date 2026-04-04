#pragma once

// Tensor
#include "synara/tensor/tensor.hpp"
#include "synara/tensor/tensor_types.hpp"

// Core
#include "synara/core/parallel.hpp"
#include "synara/core/profiler.hpp"

// Autograd
#include "synara/autograd/node.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/autograd/engine.hpp"
#include "synara/autograd/no_grad.hpp"

// Ops
#include "synara/ops/elementwise.hpp"
#include "synara/ops/reduction.hpp"
#include "synara/ops/linalg.hpp"
#include "synara/ops/convolution.hpp"
#include "synara/ops/pooling.hpp"
#include "synara/ops/activation.hpp"
#include "synara/ops/loss.hpp"
#include "synara/ops/math.hpp"

#if __has_include("synara/ops/shape.hpp")
#include "synara/ops/shape.hpp"
#endif

// NN
#include "synara/nn/module.hpp"
#include "synara/nn/parameter.hpp"
#include "synara/nn/module_utils.hpp"
#include "synara/nn/linear.hpp"
#include "synara/nn/conv2d.hpp"
#include "synara/nn/sequential.hpp"
#include "synara/nn/batch_norm.hpp"
#include "synara/nn/layer_norm.hpp"
#include "synara/nn/dropout.hpp"
#include "synara/nn/relu.hpp"
#include "synara/nn/leaky_relu.hpp"
#include "synara/nn/sigmoid.hpp"
#include "synara/nn/tanh.hpp"
#include "synara/nn/softmax.hpp"
#include "synara/nn/gelu.hpp"
#include "synara/nn/maxpool2d.hpp"
#include "synara/nn/avgpool2d.hpp"

// Optimizers
#include "synara/optim/optimizer.hpp"
#include "synara/optim/sgd.hpp"
#include "synara/optim/adam.hpp"
#include "synara/optim/rmsprop.hpp"

// Serialization
#include "synara/serialize/state_dict.hpp"

#if __has_include("synara/optim/lr_scheduler.hpp")
#include "synara/optim/lr_scheduler.hpp"
#endif

#if __has_include("synara/optim/grad_clip.hpp")
#include "synara/optim/grad_clip.hpp"
#endif

// Data
#if __has_include("synara/data/dataset.hpp")
#include "synara/data/dataset.hpp"
#endif
#if __has_include("synara/data/dataloader.hpp")
#include "synara/data/dataloader.hpp"
#endif

// Metrics
#if __has_include("synara/metrics/metrics.hpp")
#include "synara/metrics/metrics.hpp"
#endif

// Training
#if __has_include("synara/train/trainer.hpp")
#include "synara/train/trainer.hpp"
#endif
