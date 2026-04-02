#include "synara/tensor/tensor.hpp"
#include "synara/autograd/node.hpp"
#include "synara/core/error.hpp"
#include "synara/tensor/storage.hpp"
#include "synara/tensor/tensor_impl.hpp"

#include <algorithm>
#include <stdexcept>
#include <ostream>
#include <sstream>
#include <random>

namespace synara
{
    namespace
    {
        std::mt19937 &global_rng()
        {
            static std::mt19937 rng(std::random_device{}());
            return rng;
        }

        std::size_t normalize_bound(long long index, std::size_t dim_size)
        {
            long long normalized = index;
            if (normalized < 0)
            {
                normalized += static_cast<long long>(dim_size);
            }
            if (normalized < 0)
            {
                return 0;
            }
            if (normalized > static_cast<long long>(dim_size))
            {
                return dim_size;
            }
            return static_cast<std::size_t>(normalized);
        }

        std::size_t slice_length(std::size_t start, std::size_t stop, std::size_t step)
        {
            if (start >= stop)
            {
                return 0;
            }
            const std::size_t distance = stop - start;
            return (distance + step - 1) / step;
        }

    } // namespace

    Tensor::Tensor()
        : impl_(std::make_shared<TensorImpl>()) { impl_->storage->data()[0] = 0.0f; }

    Tensor::Tensor(const Shape &shape, bool requires_grad)
        : impl_(std::make_shared<TensorImpl>(shape, Strides::contiguous(shape), std::make_shared<Storage>(shape.numel(), 0.0), 0, requires_grad, true))
    {
        std::fill(impl_->storage->data(), impl_->storage->data() + impl_->storage->size(), 0.0f);
    }

    Tensor::Tensor(const Shape &shape, value_type fill_value, bool requires_grad)
        : impl_(std::make_shared<TensorImpl>(shape, Strides::contiguous(shape), std::make_shared<Storage>(shape.numel(), fill_value), 0, requires_grad, true))

    {
        std::fill(data(), data() + numel(), fill_value);
    }

    Tensor Tensor::make_view(
        Shape shape,
        Strides strides,
        std::shared_ptr<Storage> storage,
        Size offset,
        bool requires_grad,
        bool is_leaf)
    {
        return Tensor(std::make_shared<TensorImpl>(
            std::move(shape),
            std::move(strides),
            std::move(storage),
            offset,
            requires_grad,
            is_leaf));
    }

    Tensor Tensor::zeros(const Shape &shape, bool requires_grad)
    {
        return Tensor(shape, requires_grad);
    }

    Tensor Tensor::ones(const Shape &shape, bool requires_grad)
    {
        return full(shape, 1.0f, requires_grad);
    }

    Tensor Tensor::full(const Shape &shape, value_type value, bool requires_grad)
    {
        Tensor t(shape, value, requires_grad);
        std::fill(t.data(), t.data() + t.numel(), value);
        return t;
    }
    Tensor Tensor::from_vector(const Shape &shape, std::vector<value_type> values, bool requires_grad)
    {
        if (values.size() != shape.numel())
        {
            throw std::invalid_argument("from_vector values do not match shape numel");
        }
        return make_view(shape, Strides::contiguous(shape),
                         std::make_shared<Storage>(std::move(values)), 0, requires_grad, true);
    }

    Tensor Tensor::randn(const Shape &shape, value_type mean, value_type stddev, bool requires_grad)
    {
        if (stddev <= 0.0)
        {
            throw ValueError("randn(): stddev must be positive.");
        }

        std::normal_distribution<value_type> dist(mean, stddev);

        std::vector<value_type> values(shape.numel());
        for (std::size_t i = 0; i < shape.numel(); ++i)
        {
            values[i] = dist(global_rng());
        }
        return from_vector(shape, values, requires_grad);
    }

    Tensor Tensor::uniform(const Shape &shape, value_type min, value_type max, bool requires_grad)
    {
        if (min > max)
        {
            throw ValueError("uniform(): min must be <= max.");
        }

        std::uniform_real_distribution<value_type> dist(min, max);

        std::vector<value_type> values(shape.numel());
        for (std::size_t i = 0; i < shape.numel(); ++i)
        {
            values[i] = dist(global_rng());
        }
        return from_vector(shape, values, requires_grad);
    }

    void Tensor::manual_seed(std::uint64_t seed)
    {
        global_rng().seed(static_cast<std::mt19937::result_type>(seed));
    }

    std::uint64_t Tensor::random_seed()
    {
        const std::uint64_t hi = static_cast<std::uint64_t>(global_rng()());
        const std::uint64_t lo = static_cast<std::uint64_t>(global_rng()());
        return (hi << 32) | lo;
    }

    const Shape &Tensor::shape() const noexcept { return impl_->shape; }

    const Strides &Tensor::strides() const noexcept { return impl_->strides; }

    std::size_t Tensor::rank() const noexcept { return impl_->shape.rank(); }

    std::size_t Tensor::numel() const noexcept { return impl_->shape.numel(); }

    bool Tensor::is_contiguous() const { return impl_->strides.is_contiguous(impl_->shape); }
    bool Tensor::is_scalar() const noexcept { return numel() == 1; }

    bool Tensor::requires_grad() const noexcept
    {
        return impl_->requires_grad;
    }
    bool Tensor::is_leaf() const noexcept
    {
        return impl_->is_leaf;
    }

    void Tensor::set_requires_grad(bool value) noexcept
    {
        impl_->requires_grad = value;
    }

    void Tensor::set_leaf(bool value) noexcept
    {
        impl_->is_leaf = value;
    }

    const Tensor &Tensor::grad() const
    {
        if (!impl_->grad)
        {
            throw ValueError("grad(): tensor has no gradient.");
        }
        return *impl_->grad;
    }
    Tensor &Tensor::grad()
    {
        if (!impl_->grad)
        {
            throw ValueError("grad(): tensor has no gradient.");
        }
        return *impl_->grad;
    }

    void Tensor::zero_grad()
    {
        if (!impl_->grad)
        {
            return;
        }

        std::fill(impl_->grad->data(), impl_->grad->data() + impl_->grad->numel(), 0.0f);
    }
    Tensor Tensor::reshape(const Shape &new_shape) const
    {
        if (new_shape.numel() != numel())
        {
            throw std::invalid_argument("reshape changes number of elements");
        }
        if (!is_contiguous())
        {
            throw std::invalid_argument("reshape requires contiguous tensor");
        }

        return make_view(
            new_shape,
            Strides::contiguous(new_shape),
            impl_->storage,
            impl_->offset,
            requires_grad(),
            false);
    }

    Tensor Tensor::transpose(std::size_t dim0, std::size_t dim1) const
    {
        if (dim0 >= rank() || dim1 >= rank())
        {
            throw std::out_of_range("transpose dimension out of range");
        }

        std::vector<std::size_t> dims = impl_->shape.dims();
        std::vector<std::size_t> strides = impl_->strides.values();
        std::swap(dims[dim0], dims[dim1]);
        std::swap(strides[dim0], strides[dim1]);

        return make_view(
            Shape(std::move(dims)),
            Strides(std::move(strides)),
            impl_->storage,
            impl_->offset,
            impl_->requires_grad,
            impl_->is_leaf);
    }

    Tensor Tensor::flatten() const
    {
        return reshape(Shape({numel()}));
    }

    bool Tensor::has_grad() const noexcept { return impl_->grad != nullptr; }

    void Tensor::set_grad(const Tensor &grad_tensor)
    {
        if (grad_tensor.shape() != shape())
        {
            throw ShapeError("set_grad(): gradient shape must match tensor shape.");
        }

        impl_->grad = std::make_shared<Tensor>(Tensor::from_vector(shape(), std::vector<value_type>(
                                                                                grad_tensor.data(), grad_tensor.data() + grad_tensor.numel())));
    }

    void Tensor::accumulate_grad(const Tensor &grad_tensor)
    {
        if (grad_tensor.shape() != shape())
        {
            throw ShapeError("accumulate_grad(): gradient shape must match tensor shape.");
        }

        if (!impl_->grad)
        {
            impl_->grad = std::make_shared<Tensor>(Tensor::zeros(shape(), false));
        }

        for (Size i = 0; i < numel(); ++i)
        {
            impl_->grad->data()[i] += grad_tensor.data()[i];
        }
    }

    void Tensor::set_grad_fn(std::shared_ptr<Node> fn) noexcept
    {
        impl_->grad_fn = std::move(fn);
    }

    std::shared_ptr<Node> Tensor::grad_fn() const noexcept
    {
        return impl_->grad_fn;
    }

    void Tensor::backward()
    {
        if (!requires_grad())
        {
            throw ValueError("backward(): tensor does not require grad.");
        }

        if (!is_scalar())
        {
            throw ValueError("backward(): only scalar tensors supported in Milestone 3B.");
        }

        Tensor seed = Tensor::from_vector(Shape({}), {1.0f}, false);
        accumulate_grad(seed);

        if (impl_->grad_fn)
        {
            impl_->grad_fn->backward(seed);
        }
    }

    Tensor Tensor::slice(std::size_t dim, const Slice &spec) const
    {
        if (dim >= rank())
        {
            throw std::out_of_range("slice dimension out of range");
        }
        if (spec.step <= 0)
        {
            throw std::invalid_argument("slice step must be positive");
        }

        std::vector<Slice> specs(rank(), Slice::all());
        specs[dim] = spec;
        return slice(specs);
    }

    Tensor Tensor::slice(const std::vector<Slice> &specs) const
    {
        if (specs.size() > rank())
        {
            throw std::invalid_argument("too many slice specs");
        }

        std::vector<std::size_t> new_dims = impl_->shape.dims();
        std::vector<std::size_t> new_strides = impl_->strides.values();
        std::size_t new_offset = impl_->offset;

        for (std::size_t dim = 0; dim < specs.size(); ++dim)
        {
            const Slice &spec = specs[dim];
            if (spec.step <= 0)
            {
                throw std::invalid_argument("slice step must be positive");
            }

            const std::size_t dim_size = impl_->shape[dim];
            const std::size_t start =
                spec.start.has_value() ? normalize_bound(*spec.start, dim_size) : 0;
            const std::size_t stop =
                spec.stop.has_value() ? normalize_bound(*spec.stop, dim_size) : dim_size;
            const std::size_t step = static_cast<std::size_t>(spec.step);

            new_offset += start * new_strides[dim];
            new_strides[dim] *= step;
            new_dims[dim] = slice_length(start, stop, step);
        }

        return make_view(Shape(std::move(new_dims)), Strides(std::move(new_strides)), impl_->storage,
                         new_offset, impl_->requires_grad, impl_->is_leaf);
    }

    Tensor::value_type *Tensor::data() noexcept
    {
        validate_storage();
        return impl_->storage->data() + impl_->offset;
    }

    const Tensor::value_type *Tensor::data() const noexcept
    {
        validate_storage();
        return impl_->storage->data() + impl_->offset;
    }

    Tensor::value_type Tensor::item() const
    {
        if (!is_scalar())
        {
            throw ValueError("item() requires a scalar tensor.");
        }
        return data()[0];
    }

    Tensor::value_type &Tensor::at(const std::vector<std::size_t> &indices)
    {
        return impl_->storage->data()[compute_offset(indices)];
    }

    const Tensor::value_type &Tensor::at(const std::vector<std::size_t> &indices) const
    {
        return (*impl_->storage)[compute_offset(indices)];
    }

    Tensor::value_type &Tensor::operator()(std::initializer_list<std::size_t> indices)
    {
        return at(std::vector<std::size_t>(indices));
    }

    const Tensor::value_type &Tensor::operator()(
        std::initializer_list<std::size_t> indices) const
    {
        return at(std::vector<std::size_t>(indices));
    }

    Tensor::Tensor(
        std::shared_ptr<TensorImpl> impl)
        : impl_(std::move(impl)) {}

    std::size_t Tensor::compute_offset(const std::vector<std::size_t> &indices) const
    {
        if (indices.size() != rank())
        {
            throw std::invalid_argument("indices rank mismatch");
        }

        std::size_t linear = impl_->offset;
        for (std::size_t dim = 0; dim < indices.size(); ++dim)
        {
            if (indices[dim] >= impl_->shape[dim])
            {
                throw std::out_of_range("index out of bounds");
            }
            linear += indices[dim] * impl_->strides[dim];
        }

        if (linear >= impl_->storage->size())
        {
            throw std::out_of_range("computed offset out of storage bounds");
        }
        return linear;
    }

    void Tensor::validate_storage() const
    {
        if (!impl_->storage)
        {
            throw std::runtime_error("tensor storage is not initialized");
        }

        if (impl_->offset > impl_->storage->size())
        {
            throw std::out_of_range("tensor offset out of storage bounds");
        }
    }
    std::string Tensor::format_recursive(size_t dim, size_t base_offset) const
    {
        std::ostringstream oss;

        if (rank() == 0)
        {
            oss << impl_->storage->data()[impl_->offset];
            return oss.str();
        }

        if (dim == rank() - 1)
        {
            oss << "[";
            for (Dim i = 0; i < impl_->shape[dim]; ++i)
            {
                const Size idx = base_offset + static_cast<Size>(i * impl_->strides[dim]);
                oss << impl_->storage->data()[idx];
                if (i + 1 < impl_->shape[dim])
                {
                    oss << ", ";
                }
            }
            oss << "]";
            return oss.str();
        }

        oss << "[";
        for (Dim i = 0; i < impl_->shape[dim]; ++i)
        {
            const Size idx = base_offset + static_cast<Size>(i * impl_->strides[dim]);
            oss << format_recursive(dim + 1, idx);
            if (i + 1 < impl_->shape[dim])
            {
                oss << ", ";
            }
        }
        oss << "]";
        return oss.str();
    }

    std::string Tensor::to_string() const
    {
        std::ostringstream oss;
        oss << "Tensor(shape=" << impl_->shape.to_string()
            << ", strides=" << impl_->strides.to_string()
            << ", data=" << format_recursive(0, impl_->offset) << ")";
        return oss.str();
    }

    std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
    {
        os << tensor.to_string();
        return os;
    }

} // namespace synara
