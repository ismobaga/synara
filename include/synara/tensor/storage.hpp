#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace synara
{

    // Generic templated storage (works with any numeric type)
    template <typename T = double>
    class StorageBase
    {
    public:
        using value_type = T;

        StorageBase();
        explicit StorageBase(std::size_t size);
        StorageBase(std::size_t size, value_type fill_value);
        explicit StorageBase(std::vector<value_type> values);

        std::size_t size() const noexcept;

        value_type *data() noexcept;
        const value_type *data() const noexcept;

        value_type &operator[](std::size_t idx);
        const value_type &operator[](std::size_t idx) const;

        std::shared_ptr<std::vector<value_type>> shared_values() const noexcept;

    private:
        std::shared_ptr<std::vector<value_type>> values_;
    };

    // For backward compatibility: Storage defaults to double
    class Storage : public StorageBase<double>
    {
    public:
        using StorageBase<double>::StorageBase;
    };

    // Explicit float32 storage type
    using StorageFloat32 = StorageBase<float>;
    using StorageFloat64 = StorageBase<double>;

} // namespace synara

#include "synara/tensor/storage.cpp.h"
