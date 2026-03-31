#include "synara/tensor/tensor.hpp"

#include <algorithm>
#include <stdexcept>
#include <ostream>
#include <sstream>

namespace synara
{
    namespace
    {

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
        : shape_(),
          strides_(Strides::contiguous(shape_)),
          storage_(std::make_shared<Storage>(1, 0.0)),
          offset_(0) {}

    Tensor::Tensor(const Shape &shape)
        : shape_(shape),
          strides_(Strides::contiguous(shape_)),
          storage_(std::make_shared<Storage>(shape.numel(), 0.0)),
          offset_(0) {}

    Tensor::Tensor(const Shape &shape, value_type fill_value)
        : shape_(shape),
          strides_(Strides::contiguous(shape_)),
          storage_(std::make_shared<Storage>(shape.numel())),
          offset_(0)
    {
        std::fill(data(), data() + numel(), fill_value);
    }

          Tensor Tensor::zeros(const Shape &shape)
          {
              return Tensor(shape);
          }

          Tensor Tensor::ones(const Shape &shape)
          {
              return full(shape, 1.0f);
          }

          Tensor Tensor::full(const Shape &shape, value_type value)
          {
              Tensor t(shape);
              std::fill(t.data(), t.data() + t.numel(), value);
              return t;
          }
    Tensor Tensor::from_vector(const Shape &shape, std::vector<value_type> values)
    {
        if (values.size() != shape.numel())
        {
            throw std::invalid_argument("from_vector values do not match shape numel");
        }
        return Tensor(shape, Strides::contiguous(shape),
                      std::make_shared<Storage>(std::move(values)), 0);
    }

    const Shape &Tensor::shape() const noexcept { return shape_; }

    const Strides &Tensor::strides() const noexcept { return strides_; }

    std::size_t Tensor::rank() const noexcept { return shape_.rank(); }

    std::size_t Tensor::numel() const noexcept { return shape_.numel(); }

    bool Tensor::is_contiguous() const { return strides_.is_contiguous(shape_); }
    bool Tensor::is_scalar() const noexcept { return numel() == 1; }

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
        return Tensor(new_shape, Strides::contiguous(new_shape), storage_, offset_);
    }

    Tensor Tensor::transpose(std::size_t dim0, std::size_t dim1) const
    {
        if (dim0 >= rank() || dim1 >= rank())
        {
            throw std::out_of_range("transpose dimension out of range");
        }

        std::vector<std::size_t> dims = shape_.dims();
        std::vector<std::size_t> strides = strides_.values();
        std::swap(dims[dim0], dims[dim1]);
        std::swap(strides[dim0], strides[dim1]);

        return Tensor(Shape(std::move(dims)), Strides(std::move(strides)), storage_, offset_);
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

        std::vector<std::size_t> new_dims = shape_.dims();
        std::vector<std::size_t> new_strides = strides_.values();
        std::size_t new_offset = offset_;

        for (std::size_t dim = 0; dim < specs.size(); ++dim)
        {
            const Slice &spec = specs[dim];
            if (spec.step <= 0)
            {
                throw std::invalid_argument("slice step must be positive");
            }

            const std::size_t dim_size = shape_[dim];
            const std::size_t start =
                spec.start.has_value() ? normalize_bound(*spec.start, dim_size) : 0;
            const std::size_t stop =
                spec.stop.has_value() ? normalize_bound(*spec.stop, dim_size) : dim_size;
            const std::size_t step = static_cast<std::size_t>(spec.step);

            new_offset += start * new_strides[dim];
            new_strides[dim] *= step;
            new_dims[dim] = slice_length(start, stop, step);
        }

        return Tensor(Shape(std::move(new_dims)), Strides(std::move(new_strides)), storage_,
                      new_offset);
    }

    Tensor::value_type *Tensor::data() noexcept
    {
        validate_storage();
        return storage_->data() + offset_;
    }

    const Tensor::value_type *Tensor::data() const noexcept
    {
        validate_storage();
        return storage_->data() + offset_;
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
        return (*storage_)[compute_offset(indices)];
    }

    const Tensor::value_type &Tensor::at(const std::vector<std::size_t> &indices) const
    {
        return (*storage_)[compute_offset(indices)];
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
        Shape shape,
        Strides strides,
        std::shared_ptr<Storage> storage,
        std::size_t offset)
        : shape_(std::move(shape)),
          strides_(std::move(strides)),
          storage_(std::move(storage)),
          offset_(offset) {}

    std::size_t Tensor::compute_offset(const std::vector<std::size_t> &indices) const
    {
        if (indices.size() != rank())
        {
            throw std::invalid_argument("indices rank mismatch");
        }

        std::size_t linear = offset_;
        for (std::size_t dim = 0; dim < indices.size(); ++dim)
        {
            if (indices[dim] >= shape_[dim])
            {
                throw std::out_of_range("index out of bounds");
            }
            linear += indices[dim] * strides_[dim];
        }

        if (linear >= storage_->size())
        {
            throw std::out_of_range("computed offset out of storage bounds");
        }
        return linear;
    }

    void Tensor::validate_storage() const
    {
        if (!storage_)
        {
            throw std::runtime_error("tensor storage is not initialized");
        }

        if (offset_ > storage_->size())
        {
            throw std::out_of_range("tensor offset out of storage bounds");
        }
    }
    std::string Tensor::format_recursive(size_t dim, size_t base_offset) const
    {
        std::ostringstream oss;

        if (rank() == 0)
        {
            oss << storage_->data()[offset_];
            return oss.str();
        }

        if (dim == rank() - 1)
        {
            oss << "[";
            for (Dim i = 0; i < shape_[dim]; ++i)
            {
                const Size idx = base_offset + static_cast<Size>(i * strides_[dim]);
                oss << storage_->data()[idx];
                if (i + 1 < shape_[dim])
                {
                    oss << ", ";
                }
            }
            oss << "]";
            return oss.str();
        }

        oss << "[";
        for (Dim i = 0; i < shape_[dim]; ++i)
        {
            const Size idx = base_offset + static_cast<Size>(i * strides_[dim]);
            oss << format_recursive(dim + 1, idx);
            if (i + 1 < shape_[dim])
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
        oss << "Tensor(shape=" << shape_.to_string()
            << ", strides=" << strides_.to_string()
            << ", data=" << format_recursive(0, offset_) << ")";
        return oss.str();
    }

    std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
    {
        os << tensor.to_string();
        return os;
    }

} // namespace synara
