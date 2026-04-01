# Synara

Synara is a lightweight C++20 deep learning framework focused on readability,
autograd fundamentals, and end-to-end neural network building blocks.

## Current Capabilities

### Autograd Ops

- Elementwise: add, sub, mul, div
- Reduction: `sum`, `mean`
- Matrix multiplication: `matmul`
- Activations: `relu`, `sigmoid`
- Losses: `mse_loss`, `binary_cross_entropy`

### Neural Network Modules

- `Linear`
- `ReLU`
- `Sigmoid`
- `Sequential`

### Optimizer

- `SGD` with:
	- learning rate
	- momentum
	- weight decay
	- max gradient norm (clipping)

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
./build/synara_linear_regression
./build/synara_xor_mlp
```

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
