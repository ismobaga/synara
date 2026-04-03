#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "synara/data/dataset.hpp"

namespace synara
{

    class DataLoader
    {
    public:
        using Batch = std::pair<Tensor, Tensor>;

        class Iterator
        {
        public:
            using difference_type = std::ptrdiff_t;
            using value_type = Batch;
            using pointer = void;
            using reference = Batch;
            using iterator_category = std::input_iterator_tag;

            Iterator(const DataLoader *loader, std::size_t batch_index)
                : loader_(loader), batch_index_(batch_index) {}

            Batch operator*() const
            {
                return loader_->batch_at(batch_index_);
            }

            Iterator &operator++()
            {
                ++batch_index_;
                return *this;
            }

            bool operator!=(const Iterator &other) const
            {
                return batch_index_ != other.batch_index_ || loader_ != other.loader_;
            }

        private:
            const DataLoader *loader_;
            std::size_t batch_index_;
        };

        DataLoader(TensorDataset dataset, std::size_t batch_size, bool shuffle = true, std::uint64_t seed = Tensor::random_seed())
            : dataset_(std::move(dataset)), batch_size_(batch_size), shuffle_(shuffle), rng_(seed)
        {
            if (batch_size_ == 0)
            {
                throw std::invalid_argument("DataLoader: batch_size must be > 0.");
            }
            reset_indices();
        }

        std::size_t size() const noexcept
        {
            return dataset_.size();
        }

        std::size_t num_batches() const noexcept
        {
            return (size() + batch_size_ - 1) / batch_size_;
        }

        Iterator begin()
        {
            reset_indices();
            return Iterator(this, 0);
        }

        Iterator end() const
        {
            return Iterator(this, num_batches());
        }

    private:
        void reset_indices()
        {
            indices_.resize(size());
            std::iota(indices_.begin(), indices_.end(), static_cast<std::size_t>(0));
            if (shuffle_)
            {
                std::shuffle(indices_.begin(), indices_.end(), rng_);
            }
        }

        Batch batch_at(std::size_t batch_index) const
        {
            const std::size_t start = batch_index * batch_size_;
            const std::size_t stop = std::min(start + batch_size_, size());
            const std::size_t actual_batch = stop - start;
            if (actual_batch == 0)
            {
                throw std::out_of_range("DataLoader: batch index out of range");
            }

            auto first = dataset_.get(indices_[start]);
            Tensor x0 = first.first.contiguous();
            Tensor y0 = first.second.contiguous();

            std::vector<std::size_t> x_dims;
            x_dims.reserve(x0.rank() + 1);
            x_dims.push_back(actual_batch);
            for (std::size_t d : x0.shape().dims())
            {
                x_dims.push_back(d);
            }

            std::vector<std::size_t> y_dims;
            y_dims.reserve(y0.rank() + 1);
            y_dims.push_back(actual_batch);
            for (std::size_t d : y0.shape().dims())
            {
                y_dims.push_back(d);
            }

            Tensor batch_x = Tensor::zeros(Shape(std::move(x_dims)), false);
            Tensor batch_y = Tensor::zeros(Shape(std::move(y_dims)), false);

            const std::size_t x_stride = x0.numel();
            const std::size_t y_stride = y0.numel();

            std::copy(x0.data(), x0.data() + x_stride, batch_x.data());
            std::copy(y0.data(), y0.data() + y_stride, batch_y.data());

            for (std::size_t i = 1; i < actual_batch; ++i)
            {
                auto sample = dataset_.get(indices_[start + i]);
                Tensor x = sample.first.contiguous();
                Tensor y = sample.second.contiguous();

                if (x.shape() != x0.shape() || y.shape() != y0.shape())
                {
                    throw std::invalid_argument("DataLoader: all samples in a batch must have equal shape.");
                }

                std::copy(x.data(), x.data() + x_stride, batch_x.data() + i * x_stride);
                std::copy(y.data(), y.data() + y_stride, batch_y.data() + i * y_stride);
            }

            return {batch_x, batch_y};
        }

    private:
        TensorDataset dataset_;
        std::size_t batch_size_;
        bool shuffle_;
        mutable std::mt19937_64 rng_;
        std::vector<std::size_t> indices_;
    };

} // namespace synara
