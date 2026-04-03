// This file contains template implementations for Storage classes.
// It's included inline from storage.hpp to enable template instantiation.

#pragma once

#include <stdexcept>

namespace synara
{

    // Template implementations for StorageBase<T>
    template <typename T>
    StorageBase<T>::StorageBase() : values_(std::make_shared<std::vector<T>>()) {}

    template <typename T>
    StorageBase<T>::StorageBase(std::size_t size)
        : values_(std::make_shared<std::vector<T>>(size, T(0))) {}

    template <typename T>
    StorageBase<T>::StorageBase(std::size_t size, T fill_value)
        : values_(std::make_shared<std::vector<T>>(size, fill_value)) {}

    template <typename T>
    StorageBase<T>::StorageBase(std::vector<T> values)
        : values_(std::make_shared<std::vector<T>>(std::move(values))) {}

    template <typename T>
    std::size_t StorageBase<T>::size() const noexcept
    {
        return values_->size();
    }

    template <typename T>
    T *StorageBase<T>::data() noexcept
    {
        return values_->data();
    }

    template <typename T>
    const T *StorageBase<T>::data() const noexcept
    {
        return values_->data();
    }

    template <typename T>
    T &StorageBase<T>::operator[](std::size_t idx)
    {
        if (idx >= values_->size())
        {
            throw std::out_of_range("storage index out of range");
        }
        return (*values_)[idx];
    }

    template <typename T>
    const T &StorageBase<T>::operator[](std::size_t idx) const
    {
        if (idx >= values_->size())
        {
            throw std::out_of_range("storage index out of range");
        }
        return (*values_)[idx];
    }

    template <typename T>
    std::shared_ptr<std::vector<T>> StorageBase<T>::shared_values() const noexcept
    {
        return values_;
    }

} // namespace synara
