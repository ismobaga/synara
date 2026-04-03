#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iterator>

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

            Iterator(const DataLoader *loader, std::size_t batch_index);

            Batch operator*() const;

            Iterator &operator++();

            bool operator!=(const Iterator &other) const;

        private:
            const DataLoader *loader_;
            std::size_t batch_index_;
        };

        DataLoader(TensorDataset dataset, std::size_t batch_size, bool shuffle = true, std::uint64_t seed = Tensor::random_seed());
        DataLoader(std::shared_ptr<Dataset> dataset, std::size_t batch_size, bool shuffle = true, std::uint64_t seed = Tensor::random_seed());

        std::size_t size() const noexcept;

        std::size_t num_batches() const noexcept;

        Iterator begin();

        Iterator end() const;

    private:
        void reset_indices();

        Batch batch_at(std::size_t batch_index) const;

    private:
        std::shared_ptr<Dataset> dataset_;
        std::size_t batch_size_;
        bool shuffle_;
        mutable std::mt19937_64 rng_;
        std::vector<std::size_t> indices_;
    };

} // namespace synara
