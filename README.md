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

### Visualization

- Terminal ASCII line plots via `render_line_plot(...)`
- Self-contained SVG export via `write_line_plot_svg(...)`
- Works well with training-history logs and MNIST example runs

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

apply_autotuned_parallel_config(1 << 18, 4);
```

Lightweight profiling helper:

```cpp
using namespace synara;

reset_profile_data();
{
    ScopedProfile profile("train_step");
    // work to measure
}

std::cout << format_profile_summary() << "\n";
```

Trainer-level profiling hooks:

```cpp
auto stats = train_epoch_profiled(model, loader, optimizer, loss_fn);
std::cout << "loss=" << stats.mean_loss
          << ", avg_batch_ms=" << stats.average_batch_ms() << "\n";
std::cout << format_profile_summary() << "\n";
std::cout << format_profile_csv() << "\n";
std::cout << format_epoch_stats_json(stats) << "\n";

write_profile_csv("profile.csv");
write_profile_json("profile.json");
write_epoch_stats_csv(stats, "epoch.csv");
write_epoch_stats_json(stats, "epoch.json");
append_epoch_history_csv(1, "train", stats, "history.csv");
append_epoch_history_jsonl(1, "train", stats, "history.jsonl");

std::vector<PlotSeries> series = {
    {"train_loss", {0.9, 0.7, 0.5, 0.3}, '*'},
    {"eval_loss", {1.0, 0.8, 0.6, 0.4}, 'o'},
};
std::cout << render_line_plot(series, PlotOptions{.title = "Loss curves"}) << "\n";
write_line_plot_svg(series, "loss_curves.svg", PlotOptions{.title = "Loss curves"});
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
./build/synara_parallel_bench [prefix] # benchmark demo; optional `prefix` writes `<prefix>.csv/json`
./build/synara_mnist_cnn <mnist_dir> [epochs] [batch_size] [train_limit] [test_limit] [log_prefix]
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
