#include "synara/ops/shape.hpp"

#include <algorithm>
#include <stdexcept>

#include "synara/autograd/no_grad.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/core/error.hpp"

namespace synara
{

    namespace
    {
        int normalize_dim(int dim, int rank, bool allow_end = false)
        {
            if (dim < 0)
            {
                dim += rank + (allow_end ? 1 : 0);
            }
            const int hi = allow_end ? rank : (rank - 1);
            if (dim < 0 || dim > hi)
            {
                throw ShapeError("shape op: dimension out of range");
            }
            return dim;
        }

        void require_contiguous(const Tensor &t, const char *op)
        {
            if (!t.is_contiguous())
            {
                throw ShapeError(std::string(op) + "(): requires contiguous tensors.");
            }
        }
    } // namespace

    Tensor squeeze(const Tensor &a, int dim)
    {
        return a.squeeze(dim);
    }

    Tensor unsqueeze(const Tensor &a, int dim)
    {
        return a.unsqueeze(dim);
    }

    Tensor permute(const Tensor &a, const std::vector<int> &dims)
    {
        return a.permute(dims);
    }

    Tensor expand(const Tensor &a, const Shape &shape)
    {
        return a.expand(shape);
    }

    Tensor broadcast_to(const Tensor &a, const Shape &shape)
    {
        return a.broadcast_to(shape);
    }

    Tensor cat(const std::vector<Tensor> &tensors, int dim)
    {
        if (tensors.empty())
        {
            throw ValueError("cat(): tensors must not be empty.");
        }

        const auto rank = static_cast<int>(tensors.front().rank());
        dim = normalize_dim(dim, rank, false);

        const auto &base = tensors.front().shape().dims();
        std::vector<std::size_t> out_dims = base;
        out_dims[static_cast<Size>(dim)] = 0;

        bool req = false;
        for (const Tensor &t : tensors)
        {
            if (static_cast<int>(t.rank()) != rank)
            {
                throw ShapeError("cat(): all tensors must have the same rank.");
            }
            require_contiguous(t, "cat");

            const auto &dims = t.shape().dims();
            for (int i = 0; i < rank; ++i)
            {
                if (i == dim)
                {
                    continue;
                }
                if (dims[static_cast<Size>(i)] != base[static_cast<Size>(i)])
                {
                    throw ShapeError("cat(): non-concatenated dimensions must match.");
                }
            }
            out_dims[static_cast<Size>(dim)] += dims[static_cast<Size>(dim)];
            req = req || t.requires_grad();
        }

        const Size total_d = out_dims[static_cast<Size>(dim)];
        Size outer = 1;
        for (int i = 0; i < dim; ++i)
            outer *= out_dims[static_cast<Size>(i)];

        Size inner = 1;
        for (int i = dim + 1; i < rank; ++i)
            inner *= out_dims[static_cast<Size>(i)];

        Tensor out = Tensor::zeros(Shape(out_dims), false);
        Tensor::value_type *dst = out.data();

        Size offset = 0;
        for (const Tensor &t : tensors)
        {
            const Size seg = t.shape()[static_cast<Size>(dim)];
            const Tensor::value_type *src = t.data();
            for (Size o = 0; o < outer; ++o)
            {
                for (Size k = 0; k < seg; ++k)
                {
                    for (Size i = 0; i < inner; ++i)
                    {
                        dst[o * total_d * inner + (offset + k) * inner + i] =
                            src[o * seg * inner + k * inner + i];
                    }
                }
            }
            offset += seg;
        }

        req = req && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<CatNode>(tensors, dim));
        }

        return out;
    }

    Tensor stack(const std::vector<Tensor> &tensors, int dim)
    {
        if (tensors.empty())
        {
            throw ValueError("stack(): tensors must not be empty.");
        }

        const auto rank = static_cast<int>(tensors.front().rank());
        dim = normalize_dim(dim, rank, true);

        const auto &base = tensors.front().shape().dims();
        for (const Tensor &t : tensors)
        {
            if (t.shape() != tensors.front().shape())
            {
                throw ShapeError("stack(): all tensors must have identical shapes.");
            }
            require_contiguous(t, "stack");
        }

        std::vector<std::size_t> out_dims = base;
        out_dims.insert(out_dims.begin() + dim, tensors.size());

        const Size num_stacks = tensors.size();
        Size outer = 1;
        for (int i = 0; i < dim; ++i)
            outer *= out_dims[static_cast<Size>(i)];

        Size inner = 1;
        for (Size i = static_cast<Size>(dim + 1); i < out_dims.size(); ++i)
            inner *= out_dims[i];

        Tensor out = Tensor::zeros(Shape(out_dims), false);
        Tensor::value_type *dst = out.data();

        for (Size o = 0; o < outer; ++o)
        {
            for (Size s = 0; s < num_stacks; ++s)
            {
                const Tensor::value_type *src = tensors[s].data();
                for (Size i = 0; i < inner; ++i)
                {
                    dst[o * num_stacks * inner + s * inner + i] =
                        src[o * inner + i];
                }
            }
        }

        bool req = false;
        for (const Tensor &t : tensors)
        {
            req = req || t.requires_grad();
        }
        req = req && grad_mode_enabled();

        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            out.set_grad_fn(std::make_shared<StackNode>(tensors, dim));
        }

        return out;
    }

    std::vector<Tensor> split(const Tensor &a, int split_size, int dim)
    {
        if (split_size <= 0)
        {
            throw ValueError("split(): split_size must be > 0.");
        }
        require_contiguous(a, "split");

        const int rank = static_cast<int>(a.rank());
        dim = normalize_dim(dim, rank, false);

        const auto &in_dims = a.shape().dims();
        const Size total_d = in_dims[static_cast<Size>(dim)];

        Size outer = 1;
        for (int i = 0; i < dim; ++i)
            outer *= in_dims[static_cast<Size>(i)];

        Size inner = 1;
        for (int i = dim + 1; i < rank; ++i)
            inner *= in_dims[static_cast<Size>(i)];

        std::vector<Tensor> out;
        out.reserve((total_d + static_cast<Size>(split_size) - 1) / static_cast<Size>(split_size));

        const Tensor::value_type *src = a.data();
        Size offset = 0;
        while (offset < total_d)
        {
            const Size chunk = std::min<Size>(static_cast<Size>(split_size), total_d - offset);
            std::vector<std::size_t> part_dims = in_dims;
            part_dims[static_cast<Size>(dim)] = chunk;
            Tensor piece = Tensor::zeros(Shape(part_dims), false);
            Tensor::value_type *dst = piece.data();

            for (Size o = 0; o < outer; ++o)
            {
                for (Size k = 0; k < chunk; ++k)
                {
                    for (Size i = 0; i < inner; ++i)
                    {
                        dst[o * chunk * inner + k * inner + i] =
                            src[o * total_d * inner + (offset + k) * inner + i];
                    }
                }
            }

            const bool req = a.requires_grad() && grad_mode_enabled();
            piece.set_requires_grad(req);
            piece.set_leaf(!req);
            if (req)
            {
                piece.set_grad_fn(std::make_shared<SplitPieceNode>(a, dim, offset, chunk));
            }

            out.push_back(piece);
            offset += chunk;
        }

        return out;
    }

} // namespace synara
