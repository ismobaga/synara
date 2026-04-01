#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "synara/core/types.hpp"
#include "synara/core/error.hpp"
#include "synara/tensor/shape.hpp"
#include "synara/tensor/slice.hpp"
#include "synara/tensor/storage.hpp"
#include "synara/tensor/strides.hpp"

namespace synara
{

    class Tensor
    {
    public:
        using value_type = Storage::value_type;

        Tensor();
        explicit Tensor(const Shape &shape);
        Tensor(const Shape &shape, value_type fill_value);

        static Tensor zeros(const Shape &shape);
        static Tensor ones(const Shape &shape);
        static Tensor full(const Shape &shape, value_type value);
        static Tensor from_vector(const Shape &shape, std::vector<value_type> values);

        const Shape &shape() const noexcept;
        const Strides &strides() const noexcept;

        std::size_t rank() const noexcept;
        std::size_t numel() const noexcept;
        bool is_contiguous() const;
        bool is_scalar() const noexcept;

        value_type *data() noexcept;
        const value_type *data() const noexcept;

        Tensor reshape(const Shape &new_shape) const;
        Tensor transpose(std::size_t dim0, std::size_t dim1) const;
        Tensor flatten() const;

        Tensor slice(std::size_t dim, const Slice &spec) const;
        Tensor slice(const std::vector<Slice> &specs) const;

        value_type item() const;


        value_type &at(const std::vector<std::size_t> &indices);
        const value_type &at(const std::vector<std::size_t> &indices) const;

        value_type &operator()(std::initializer_list<std::size_t> indices);
        const value_type &operator()(std::initializer_list<std::size_t> indices) const;

        std::string to_string() const;


    private:
        Tensor(
            Shape shape,
            Strides strides,
            std::shared_ptr<Storage> storage,
            std::size_t offset);

        std::size_t compute_offset(const std::vector<std::size_t> &indices) const;

        void validate_storage() const;

        std::string format_recursive(size_t dim, size_t base_offset) const;

        Shape shape_;
        Strides strides_;
        std::shared_ptr<Storage> storage_;
        std::size_t offset_ = 0;
    };

    std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

} // namespace synara
