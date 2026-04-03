#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "synara/tensor/slice.hpp"
#include "synara/tensor/tensor.hpp"

namespace synara
{

    class Dataset
    {
    public:
        virtual ~Dataset() = default;
        virtual std::size_t len() const noexcept = 0;
        virtual std::pair<Tensor, Tensor> get(std::size_t index) const = 0;
    };

    class TensorDataset : public Dataset
    {
    public:
        TensorDataset(Tensor inputs, Tensor targets);

        std::size_t len() const noexcept override;
        std::size_t size() const noexcept;

        std::pair<Tensor, Tensor> get(std::size_t index) const override;

    private:
        Tensor inputs_;
        Tensor targets_;
    };

} // namespace synara
