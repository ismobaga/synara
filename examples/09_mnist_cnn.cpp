#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "synara/nn/avgpool2d.hpp"
#include "synara/nn/conv2d.hpp"
#include "synara/nn/maxpool2d.hpp"
#include "synara/nn/relu.hpp"
#include "synara/nn/sequential.hpp"
#include "synara/nn/softmax.hpp"
#include "synara/ops/loss.hpp"
#include "synara/optim/adam.hpp"
#include "synara/serialize/state_dict.hpp"
#include "synara/tensor/slice.hpp"

namespace
{

    uint32_t read_be_u32(std::ifstream &in)
    {
        unsigned char bytes[4]{};
        in.read(reinterpret_cast<char *>(bytes), 4);
        if (!in)
        {
            throw std::runtime_error("Failed to read 4-byte big-endian integer.");
        }
        return (static_cast<uint32_t>(bytes[0]) << 24) |
               (static_cast<uint32_t>(bytes[1]) << 16) |
               (static_cast<uint32_t>(bytes[2]) << 8) |
               (static_cast<uint32_t>(bytes[3]));
    }

    synara::Tensor load_idx_images(const std::string &path, std::size_t limit)
    {
        using namespace synara;

        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error("Cannot open images file: " + path);
        }

        const uint32_t magic = read_be_u32(in);
        const uint32_t count = read_be_u32(in);
        const uint32_t rows = read_be_u32(in);
        const uint32_t cols = read_be_u32(in);

        if (magic != 2051)
        {
            throw std::runtime_error("Invalid images IDX magic for file: " + path);
        }
        if (rows != 28 || cols != 28)
        {
            throw std::runtime_error("Expected 28x28 MNIST images.");
        }

        const std::size_t n = std::min<std::size_t>(count, limit);
        std::vector<Tensor::value_type> values(n * rows * cols);

        for (std::size_t i = 0; i < n * rows * cols; ++i)
        {
            unsigned char pixel = 0;
            in.read(reinterpret_cast<char *>(&pixel), 1);
            if (!in)
            {
                throw std::runtime_error("Unexpected EOF while reading images: " + path);
            }
            values[i] = static_cast<Tensor::value_type>(pixel) / 255.0f;
        }

        return Tensor::from_vector(Shape({n, 1, rows, cols}), std::move(values), false);
    }

    std::vector<int> load_idx_labels(const std::string &path, std::size_t limit)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error("Cannot open labels file: " + path);
        }

        const uint32_t magic = read_be_u32(in);
        const uint32_t count = read_be_u32(in);

        if (magic != 2049)
        {
            throw std::runtime_error("Invalid labels IDX magic for file: " + path);
        }

        const std::size_t n = std::min<std::size_t>(count, limit);
        std::vector<int> labels(n);

        for (std::size_t i = 0; i < n; ++i)
        {
            unsigned char label = 0;
            in.read(reinterpret_cast<char *>(&label), 1);
            if (!in)
            {
                throw std::runtime_error("Unexpected EOF while reading labels: " + path);
            }
            labels[i] = static_cast<int>(label);
        }

        return labels;
    }

    synara::Tensor one_hot_4d(const std::vector<int> &labels, int classes)
    {
        using namespace synara;

        std::vector<Tensor::value_type> values(labels.size() * static_cast<std::size_t>(classes), 0.0f);
        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            const int c = labels[i];
            if (c < 0 || c >= classes)
            {
                throw std::runtime_error("Label out of range while creating one-hot tensor.");
            }
            values[i * static_cast<std::size_t>(classes) + static_cast<std::size_t>(c)] = 1.0f;
        }

        return Tensor::from_vector(Shape({labels.size(), static_cast<std::size_t>(classes), 1, 1}),
                                   std::move(values), false);
    }

    int argmax_channel_10(const synara::Tensor &probs, std::size_t n)
    {
        int best = 0;
        float best_value = probs.at({n, 0, 0, 0});
        for (int c = 1; c < 10; ++c)
        {
            const float v = probs.at({n, static_cast<std::size_t>(c), 0, 0});
            if (v > best_value)
            {
                best_value = v;
                best = c;
            }
        }
        return best;
    }

    float accuracy(const synara::Tensor &probs, const std::vector<int> &labels)
    {
        std::size_t correct = 0;
        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            if (argmax_channel_10(probs, i) == labels[i])
            {
                ++correct;
            }
        }

        return static_cast<float>(correct) / static_cast<float>(labels.size());
    }

} // namespace

int main(int argc, char **argv)
{
    using namespace synara;

    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <mnist_dir> [epochs] [batch_size] [train_limit] [test_limit]\n";
        std::cout << "Expected files in <mnist_dir>:\n";
        std::cout << "  train-images-idx3-ubyte\n";
        std::cout << "  train-labels-idx1-ubyte\n";
        std::cout << "  t10k-images-idx3-ubyte\n";
        std::cout << "  t10k-labels-idx1-ubyte\n";
        return 1;
    }

    const std::string data_dir = argv[1];
    const int epochs = (argc > 2) ? std::max(1, std::stoi(argv[2])) : 5;
    const std::size_t batch_size = (argc > 3) ? std::max(1, std::stoi(argv[3])) : 64;
    const std::size_t train_limit = (argc > 4) ? static_cast<std::size_t>(std::stoll(argv[4])) : 6000;
    const std::size_t test_limit = (argc > 5) ? static_cast<std::size_t>(std::stoll(argv[5])) : 1000;

    const std::string train_images_path = data_dir + "/train-images.idx3-ubyte";
    const std::string train_labels_path = data_dir + "/train-labels.idx1-ubyte";
    const std::string test_images_path = data_dir + "/t10k-images.idx3-ubyte";
    const std::string test_labels_path = data_dir + "/t10k-labels.idx1-ubyte";

    try
    {
        Tensor train_x = load_idx_images(train_images_path, train_limit);
        std::vector<int> train_labels = load_idx_labels(train_labels_path, train_limit);
        Tensor train_y = one_hot_4d(train_labels, 10);

        Tensor test_x = load_idx_images(test_images_path, test_limit);
        std::vector<int> test_labels = load_idx_labels(test_labels_path, test_limit);
        Tensor test_y = one_hot_4d(test_labels, 10);

        auto conv1 = std::make_shared<Conv2d>(1, 8, 3, 3, 1, 1, 1, 1, true);
        auto relu1 = std::make_shared<ReLU>();
        auto pool1 = std::make_shared<MaxPool2d>(2, 2, 2, 2, 0, 0);
        auto conv2 = std::make_shared<Conv2d>(8, 10, 3, 3, 1, 1, 1, 1, true);
        auto gap = std::make_shared<AvgPool2d>(14, 14, 14, 14, 0, 0);
        auto softmax_head = std::make_shared<Softmax>(1);

        Sequential model({conv1, relu1, pool1, conv2, gap, softmax_head});

        std::vector<Parameter *> params = model.parameters();
        std::vector<Tensor *> param_tensors;
        param_tensors.reserve(params.size());
        for (Parameter *p : params)
        {
            param_tensors.push_back(&p->tensor());
        }

        Adam optimizer(param_tensors, AdamOptions{.lr = 0.001, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8, .weight_decay = 0.0});

        std::cout << "train samples: " << train_x.shape()[0] << ", test samples: " << test_x.shape()[0] << "\n";
        std::cout << "epochs=" << epochs << ", batch_size=" << batch_size << "\n\n";

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float epoch_loss_sum = 0.0f;
            std::size_t seen = 0;

            for (std::size_t start = 0; start < train_x.shape()[0]; start += batch_size)
            {
                const std::size_t end = std::min<std::size_t>(start + batch_size, train_x.shape()[0]);
                const Slice batch_slice{static_cast<long long>(start), static_cast<long long>(end), 1};

                Tensor x_batch = train_x.slice(0, batch_slice);
                Tensor y_batch = train_y.slice(0, batch_slice);

                optimizer.zero_grad();
                Tensor pred = model(x_batch);
                Tensor loss = mse_loss(pred, y_batch);
                loss.backward();
                optimizer.step();

                const std::size_t batch_n = end - start;
                epoch_loss_sum += loss.item() * static_cast<float>(batch_n);
                seen += batch_n;
            }

            Tensor train_pred = model(train_x);
            Tensor test_pred = model(test_x);
            const float train_loss = mse_loss(train_pred, train_y).item();
            const float test_loss = mse_loss(test_pred, test_y).item();
            const float train_acc = accuracy(train_pred, train_labels);
            const float test_acc = accuracy(test_pred, test_labels);

            std::cout << "epoch " << (epoch + 1)
                      << " train_loss=" << train_loss
                      << " test_loss=" << test_loss
                      << " train_acc=" << train_acc
                      << " test_acc=" << test_acc << "\n";

            (void)epoch_loss_sum;
            (void)seen;
        }

        const std::string checkpoint = "mnist_cnn_checkpoint.syn";
        save_state_dict(model.state_dict(), checkpoint);
        std::cout << "\nSaved checkpoint: " << checkpoint << "\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
