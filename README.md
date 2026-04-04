# Synara

Synara is a lightweight C++20 deep learning framework focused on readability,
autograd fundamentals, and end-to-end neural network building blocks.

## Current Capabilities

### Tensor

- Shape management, strides, contiguous memory layout
- Creation: `zeros`, `ones`, `full`, `from_vector`, `randn`, `uniform`
- Views: `reshape`, `transpose`, `flatten`, `slice`
- Reproducibility: `Tensor::manual_seed()`, `Tensor::random_seed()`

### Autograd Ops

- Elementwise: `add`, `sub`, `mul`, `div`
- Reduction: `sum`, `mean`
- Matrix multiplication: `matmul`
- Activations: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax`, `gelu`
- Losses: `mse_loss`, `binary_cross_entropy`, `cross_entropy_loss`

### Neural Network Modules

- `Linear` — fully-connected layer with optional bias
- `Conv2d` — 2-D convolution with stride, padding, dilation, groups
- `MaxPool2d`, `AvgPool2d` — spatial pooling
- `BatchNorm1d`, `BatchNorm2d` — batch normalisation with running stats
- `LayerNorm` — layer normalisation
- `Dropout` — inverted-dropout with train/eval modes
- `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`, `GELU` — activation wrappers
- `Sequential` — ordered module container

### Optimizers

- `SGD` — with momentum, weight decay, and gradient-norm clipping
- `Adam` — with configurable β₁/β₂, ε, and weight decay
- `RMSprop` — with momentum, α (smoothing constant), ε, and weight decay

### Serialization

- Module `state_dict()` and `load_state_dict()`
- File checkpoint save/load with versioned text format (`SYNARA_STATE_V1`)

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

Optional threaded kernels via OpenMP:

```bash
cmake -S . -B build -DSYNARA_ENABLE_OPENMP=ON
cmake --build build -j
```

Runtime thread/tuning control:

```cpp
using namespace synara;

set_num_threads(4);
ParallelConfig tuned = parallel_config();
tuned.conv2d_threshold = 1 << 15;
tuned.linear_threshold = 1 << 15;
set_parallel_config(tuned);
```

## Run Tests

```bash
cd build
ctest --output-on-failure
```

The suite includes tensor, autograd, nn, optimizer, serialization, and
finite-difference gradient-validation tests (56+ test files).

## Run Examples

```bash
./build/synara                         # tensor basics
./build/synara_linear_regression       # linear regression with SGD
./build/synara_xor_mlp                 # XOR problem with a 2-layer MLP
./build/synara_conv2d_basics           # 2-D convolution
./build/synara_pooling_basics          # max pooling
./build/synara_avg_pooling_basics      # average pooling
./build/synara_conv2d_advanced         # groups & dilation
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

Current implementation is stable and covered by automated tests, including
numerical finite-difference checks for key gradients.
