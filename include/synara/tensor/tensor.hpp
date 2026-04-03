#pragma once

#include <cstdint>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "synara/core/types.hpp"
#include "synara/core/error.hpp"
#include "synara/autograd/node.hpp"
#include "synara/tensor/shape.hpp"
#include "synara/tensor/strides.hpp"
#include "synara/tensor/storage.hpp"
#include "synara/tensor/slice.hpp"

#include "synara/tensor/tensor_impl.hpp"

namespace synara
{

    class Tensor
    {
    public:
        using value_type = Storage::value_type;

        Tensor();
        explicit Tensor(const Shape &shape, bool requires_grad = false);
        Tensor(const Shape &shape, value_type fill_value, bool requires_grad = false);

        static Tensor zeros(const Shape &shape, bool requires_grad = false);
        static Tensor ones(const Shape &shape, bool requires_grad = false);
        static Tensor full(const Shape &shape, value_type value, bool requires_grad = false);
        static Tensor from_vector(const Shape &shape, std::vector<value_type> values, bool requires_grad = false);
        static Tensor randn(const Shape &shape, value_type mean = 0.0f, value_type stddev = 1.0f, bool requires_grad = false);
        static Tensor uniform(const Shape &shape, value_type min = 0.0f, value_type max = 1.0f, bool requires_grad = false);
        static void manual_seed(std::uint64_t seed);
        static std::uint64_t random_seed();

        const Shape &shape() const noexcept;
        const Strides &strides() const noexcept;

        std::size_t rank() const noexcept;
        std::size_t numel() const noexcept;
        bool is_contiguous() const;
        bool is_scalar() const noexcept;
        // Autograd API
        bool requires_grad() const noexcept;
        bool is_leaf() const noexcept;
        bool has_grad() const noexcept;
        void accumulate_grad(const Tensor &grad);
        const Tensor &grad() const;
        Tensor &grad();
        void set_grad(const Tensor &grad_tensor);
        void set_requires_grad(bool value) noexcept;

        void set_leaf(bool value) noexcept;
        void zero_grad();

        void backward(bool retain_graph = false);

        value_type *data() noexcept;
        const value_type *data() const noexcept;

        Tensor detach() const;

        Tensor reshape(const Shape &new_shape) const;
        Tensor transpose(std::size_t dim0, std::size_t dim1) const;
        Tensor flatten() const;

        Tensor squeeze(int dim = -1) const;
        Tensor unsqueeze(int dim) const;
        Tensor permute(const std::vector<int> &dims) const;
        Tensor expand(const Shape &shape) const;
        Tensor broadcast_to(const Shape &shape) const;
        Tensor contiguous() const;

        Tensor clone() const;

        Tensor slice(std::size_t dim, const Slice &spec) const;
        Tensor slice(const std::vector<Slice> &specs) const;

        value_type item() const;

        value_type &at(const std::vector<std::size_t> &indices);
        const value_type &at(const std::vector<std::size_t> &indices) const;

        value_type &operator()(std::initializer_list<std::size_t> indices);
        const value_type &operator()(std::initializer_list<std::size_t> indices) const;

        std::string to_string() const;

        void set_grad_fn(std::shared_ptr<Node> fn) noexcept;
        std::shared_ptr<Node> grad_fn() const noexcept;

    private:
        explicit Tensor(std::shared_ptr<TensorImpl> impl);

        static Tensor make_view(
            Shape shape,
            Strides strides,
            std::shared_ptr<Storage> storage,
            Size offset,
            bool requires_grad,
            bool is_leaf);

        std::size_t compute_offset(const std::vector<std::size_t> &indices) const;

        void validate_storage() const;

        std::string format_recursive(size_t dim, size_t base_offset) const;

    private:
        std::shared_ptr<TensorImpl> impl_;
    };

    std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

} // namespace synara
