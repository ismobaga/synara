# Synara

Synara is a lightweight C++20 deep learning framework focused on readability,
autograd fundamentals, and end-to-end neural network building blocks.

## Current Capabilities

### Tensor System

- Multi-dimensional tensors with arbitrary shapes and strides
- Views (reshape, transpose, flatten, slice) with zero-copy semantics
- Random initialisation: `randn()`, `uniform()`
- Deterministic seeding: `manual_seed()`, `random_seed()`

### Autograd Ops

- Elementwise: `add`, `sub`, `mul`, `div`
- Reduction: `sum`, `mean`
- Matrix multiplication: `matmul`
- Activations: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax`
- Losses: `mse_loss`, `binary_cross_entropy`
- Convolution: `conv2d` (stride, padding, dilation, groups)
- Pooling: `max_pool2d`, `avg_pool2d`

### Neural Network Modules

- `Linear`
- `Conv2d`
- `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`
- `BatchNorm` (batch normalisation with running statistics)
- `LayerNorm`
- `Dropout`
- `MaxPool2d`, `AvgPool2d`
- `Sequential`

### Optimizers

- `SGD` with:
  - learning rate
  - momentum
  - weight decay
  - max gradient norm (clipping)
- `Adam` with:
  - learning rate, β₁, β₂, ε
  - weight decay

### Serialization

- Module `state_dict()` and `load_state_dict()`
- File checkpoint save/load with versioned text format (`SYNARA_STATE_V1`)

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run Tests

```bash
cd build
ctest --output-on-failure
```

The suite includes tensor, autograd, nn, optimizer, serialization, and finite-difference gradient validation tests.

## Run Examples

```bash
./build/synara
./build/synara_tensor_basics
./build/synara_linear_regression
./build/synara_xor_mlp
./build/synara_mnist_cnn <mnist_dir> [epochs] [batch_size] [train_limit] [test_limit]
```

For the MNIST example, place these IDX files in `<mnist_dir>`:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

## Minimal Training Sketch

```cpp
using namespace synara;

Linear model(1, 1, true);
SGD optim(model.parameters(), SGDOptions{.lr = 0.05f});

Tensor x = Tensor::from_vector(Shape({4, 1}), {1, 2, 3, 4});
Tensor y = Tensor::from_vector(Shape({4, 1}), {3, 5, 7, 9});

for (int epoch = 0; epoch < 300; ++epoch) {
    Tensor pred = model(x);
    Tensor loss = mse_loss(pred, y);

    optim.zero_grad();
    loss.backward();
    optim.step();
}
```

## Checkpoint Save/Load

```cpp
using namespace synara;

auto state = model.state_dict();
save_state_dict(state, "checkpoint.syn");

auto loaded = load_state_dict("checkpoint.syn");
model.load_state_dict(loaded);
```

## Status

Current implementation is stable and covered by automated tests, including numerical finite-difference checks for key gradients.
