#include "synara/data/dataset.hpp"

namespace synara
{

    TensorDataset::TensorDataset(Tensor inputs, Tensor targets)
        : inputs_(std::move(inputs)), targets_(std::move(targets))
    {
        if (inputs_.rank() == 0 || targets_.rank() == 0)
        {
            throw std::invalid_argument("TensorDataset: inputs and targets must have batch dimension.");
        }
        if (inputs_.shape()[0] != targets_.shape()[0])
        {
            throw std::invalid_argument("TensorDataset: inputs and targets batch dimension mismatch.");
        }
    }

    std::size_t TensorDataset::size() const noexcept
    {
        return inputs_.shape()[0];
    }

    std::pair<Tensor, Tensor> TensorDataset::get(std::size_t index) const
    {
        if (index >= size())
        {
            throw std::out_of_range("TensorDataset::get index out of range");
        }

        Slice one;
        one.start = static_cast<long long>(index);
        one.stop = static_cast<long long>(index + 1);
        one.step = 1;

        Tensor x = inputs_.slice(0, one);
        Tensor y = targets_.slice(0, one);
        if (x.rank() > 0)
        {
            x = x.squeeze(0);
        }
        if (y.rank() > 0)
        {
            y = y.squeeze(0);
        }
        return {x, y};
    }

} // namespace synara
