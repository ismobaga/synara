# Synara

Synara is a lightweight C++20 deep learning framework focused on readability,
autograd fundamentals, and end-to-end neural network building blocks.

## Current Capabilities

### Tensor Core

- Shape and stride system
- Shared storage-backed tensor views
- Indexing and slicing
- Reshape, transpose, flatten
- Random tensor factory:
	- `Tensor::uniform(shape, min, max)`
	- `Tensor::randn(shape, mean, stddev)`

### Autograd + Ops

- Elementwise: add, sub, mul, div
- Reductions: sum, mean
- Linalg: matmul
- Activations:
	- relu
	- leaky_relu
	- sigmoid
	- tanh
	- softmax (numerically stable)
- Losses:
	- mse_loss
	- binary_cross_entropy

### Neural Network Modules

- Linear
- Conv2d (stride, padding, dilation, groups, bias)
- MaxPool2d
- AvgPool2d
- BatchNorm1d
- BatchNorm2d
- Dropout
- ReLU, Sigmoid, Tanh, LeakyReLU, Softmax
- Sequential

### Optimization + Serialization

- SGD (momentum, weight decay, max grad norm)
- Adam (beta1/beta2, eps, weight decay, bias correction)
- `state_dict` / `load_state_dict` support for modules

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

## Run Examples

Examples are built as standalone executables.

```bash
./build/synara_tensor_basics
./build/synara_ops_basics
./build/synara_autograd_basics
./build/synara_linear_regression
./build/synara_xor_mlp
./build/synara_conv2d_basics
./build/synara_pooling_basics
./build/synara_avg_pooling_basics
./build/synara_conv2d_advanced
```

## Project Layout

- `include/synara/`: public headers
- `src/`: implementations
- `tests/`: unit and finite-difference gradient tests
- `examples/`: usage demos
- `tools/`: CLI entrypoint
