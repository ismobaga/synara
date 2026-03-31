#include "synara/tensor/storage.hpp"

#include <stdexcept>

namespace synara
{

    Storage::Storage() : values_(std::make_shared<std::vector<value_type>>()) {}

    Storage::Storage(std::size_t size)
        : values_(std::make_shared<std::vector<value_type>>(size, 0.0)) {}

    Storage::Storage(std::vector<value_type> values)
        : values_(std::make_shared<std::vector<value_type>>(std::move(values))) {}

    std::size_t Storage::size() const noexcept { return values_->size(); }

    Storage::value_type *Storage::data() noexcept { return values_->data(); }

    const Storage::value_type *Storage::data() const noexcept { return values_->data(); }

    Storage::value_type &Storage::operator[](std::size_t idx)
    {
        if (idx >= values_->size())
        {
            throw std::out_of_range("storage index out of range");
        }
        return (*values_)[idx];
    }

    const Storage::value_type &Storage::operator[](std::size_t idx) const
    {
        if (idx >= values_->size())
        {
            throw std::out_of_range("storage index out of range");
        }
        return (*values_)[idx];
    }

    std::shared_ptr<std::vector<Storage::value_type>> Storage::shared_values() const noexcept
    {
        return values_;
    }

} // namespace synara
