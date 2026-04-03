#include "synara/ops/reduction.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/core/error.hpp"
#include "synara/autograd/no_grad.hpp"

namespace synara
{

    static void require_contiguous(const Tensor &t, const char *op_name)
    {
        if (!t.is_contiguous())
        {
            throw ValueError(std::string(op_name) + ": requires contiguous tensor in current milestone.");
        }
    }

    Tensor sum(const Tensor &a)
    {
        require_contiguous(a, "sum");
        Tensor::value_type total = 0;

        for (Size i = 0; i < a.numel(); ++i)
        {
            total += a.data()[i];
        }

        bool req = a.requires_grad() && grad_mode_enabled();
        Tensor out = Tensor::from_vector(Shape({}), {total}, req);
        out.set_leaf(!req);
        if (req)
        {

            auto node = std::make_shared<SumNode>(a);
            out.set_grad_fn(node);
        }
        return out;
    }

    Tensor mean(const Tensor &a)
    {
        require_contiguous(a, "mean");
        Tensor::value_type total = 0;

        for (Size i = 0; i < a.numel(); ++i)
        {
            total += a.data()[i];
        }

        Tensor out = Tensor::from_vector(Shape({}), {total / static_cast<Tensor::value_type>(a.numel())}, a.requires_grad());
        out.set_leaf(!a.requires_grad());
        if (a.requires_grad())
        {
            auto node = std::make_shared<MeanNode>(a);
            out.set_grad_fn(node);
        }
        return out;
    }

    static int normalize_dim(int dim, int rank)
    {
        if (dim < 0)
            dim += rank;
        if (dim < 0 || dim >= rank)
            throw ValueError("reduction: dim out of range");
        return dim;
    }
    Tensor sum(const Tensor &a, int dim, bool keepdim)
    {
        require_contiguous(a, "sum");
        int r = static_cast<int>(a.rank());
        dim = normalize_dim(dim, r);
        const auto &in_dims = a.shape().dims();
        Size dim_size = in_dims[dim];
        // Compute output shape
        std::vector<std::size_t> out_dims;
        for (int i = 0; i < r; ++i)
        {
            if (i == dim)
            {
                if (keepdim)
                    out_dims.push_back(1);
            }
            else
            {
                out_dims.push_back(in_dims[i]);
            }
        }
        if (out_dims.empty())
            out_dims.push_back(1); // shouldn't happen for rank>=1
        Size outer = 1;
        for (int i = 0; i < dim; ++i)
            outer *= in_dims[i];
        Size inner = 1;
        for (int i = dim + 1; i < r; ++i)
            inner *= in_dims[i];
        Shape out_shape(out_dims);
        Tensor out = Tensor::zeros(out_shape, false);
        Tensor::value_type *odata = out.data();
        const Tensor::value_type *idata = a.data();
        for (Size o = 0; o < outer; ++o)
        {
            for (Size i = 0; i < inner; ++i)
            {
                Tensor::value_type acc = 0;
                for (Size d = 0; d < dim_size; ++d)
                {
                    acc += idata[o * dim_size * inner + d * inner + i];
                }
                odata[o * inner + i] = acc;
            }
        }
        bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            auto node = std::make_shared<SumDimNode>(a, dim, keepdim);
            out.set_grad_fn(node);
        }
        return out;
    }

    Tensor max(const Tensor &a, int dim, bool keepdim)
    {
        require_contiguous(a, "max");
        int r = static_cast<int>(a.rank());
        dim = normalize_dim(dim, r);
        const auto &in_dims = a.shape().dims();
        Size dim_size = in_dims[dim];
        std::vector<std::size_t> out_dims;
        for (int i = 0; i < r; ++i)
        {
            if (i == dim)
            {
                if (keepdim)
                    out_dims.push_back(1);
            }
            else
            {
                out_dims.push_back(in_dims[i]);
            }
        }
        if (out_dims.empty())
            out_dims.push_back(1);
        Size outer = 1;
        for (int i = 0; i < dim; ++i)
            outer *= in_dims[i];
        Size inner = 1;
        for (int i = dim + 1; i < r; ++i)
            inner *= in_dims[i];
        Shape out_shape(out_dims);
        Tensor out = Tensor::zeros(out_shape, false);
        Tensor::value_type *odata = out.data();
        const Tensor::value_type *idata = a.data();
        for (Size o = 0; o < outer; ++o)
        {
            for (Size i = 0; i < inner; ++i)
            {
                Tensor::value_type best = idata[o * dim_size * inner + i];
                for (Size d = 1; d < dim_size; ++d)
                {
                    Tensor::value_type v = idata[o * dim_size * inner + d * inner + i];
                    if (v > best)
                        best = v;
                }
                odata[o * inner + i] = best;
            }
        }
        bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            auto node = std::make_shared<MaxDimNode>(a, dim, keepdim, out);
            out.set_grad_fn(node);
        }
        return out;
    }
    Tensor min(const Tensor &a, int dim, bool keepdim)
    {
        require_contiguous(a, "min");
        int r = static_cast<int>(a.rank());
        dim = normalize_dim(dim, r);
        const auto &in_dims = a.shape().dims();
        Size dim_size = in_dims[dim];
        std::vector<std::size_t> out_dims;
        for (int i = 0; i < r; ++i)
        {
            if (i == dim)
            {
                if (keepdim)
                    out_dims.push_back(1);
            }
            else
            {
                out_dims.push_back(in_dims[i]);
            }
        }
        if (out_dims.empty())
            out_dims.push_back(1);
        Size outer = 1;
        for (int i = 0; i < dim; ++i)
            outer *= in_dims[i];
        Size inner = 1;
        for (int i = dim + 1; i < r; ++i)
            inner *= in_dims[i];
        Shape out_shape(out_dims);
        Tensor out = Tensor::zeros(out_shape, false);
        Tensor::value_type *odata = out.data();
        const Tensor::value_type *idata = a.data();
        for (Size o = 0; o < outer; ++o)
        {
            for (Size i = 0; i < inner; ++i)
            {
                Tensor::value_type best = idata[o * dim_size * inner + i];
                for (Size d = 1; d < dim_size; ++d)
                {
                    Tensor::value_type v = idata[o * dim_size * inner + d * inner + i];
                    if (v < best)
                        best = v;
                }
                odata[o * inner + i] = best;
            }
        }
        bool req = a.requires_grad() && grad_mode_enabled();
        out.set_requires_grad(req);
        out.set_leaf(!req);
        if (req)
        {
            auto node = std::make_shared<MinDimNode>(a, dim, keepdim, out);
            out.set_grad_fn(node);
        }
        return out;
    }
    Tensor argmax(const Tensor &a, int dim, bool keepdim)
    {
        require_contiguous(a, "argmax");
        int r = static_cast<int>(a.rank());
        dim = normalize_dim(dim, r);
        const auto &in_dims = a.shape().dims();
        Size dim_size = in_dims[dim];
        std::vector<std::size_t> out_dims;
        for (int i = 0; i < r; ++i)
        {
            if (i == dim)
            {
                if (keepdim)
                    out_dims.push_back(1);
            }
            else
            {
                out_dims.push_back(in_dims[i]);
            }
        }
        if (out_dims.empty())
            out_dims.push_back(1);
        Size outer = 1;
        for (int i = 0; i < dim; ++i)
            outer *= in_dims[i];
        Size inner = 1;
        for (int i = dim + 1; i < r; ++i)
            inner *= in_dims[i];
        Shape out_shape(out_dims);
        Tensor out = Tensor::zeros(out_shape, false);
        Tensor::value_type *odata = out.data();
        const Tensor::value_type *idata = a.data();
        for (Size o = 0; o < outer; ++o)
        {
            for (Size i = 0; i < inner; ++i)
            {
                Tensor::value_type best = idata[o * dim_size * inner + i];
                Size best_idx = 0;
                for (Size d = 1; d < dim_size; ++d)
                {
                    Tensor::value_type v = idata[o * dim_size * inner + d * inner + i];
                    if (v > best)
                    {
                        best = v;
                        best_idx = d;
                    }
                }
                odata[o * inner + i] = static_cast<Tensor::value_type>(best_idx);
            }
        }
        // argmax is non-differentiable
        return out;
    }
    Tensor argmin(const Tensor &a, int dim, bool keepdim)
    {
        require_contiguous(a, "argmin");
        int r = static_cast<int>(a.rank());
        dim = normalize_dim(dim, r);
        const auto &in_dims = a.shape().dims();
        Size dim_size = in_dims[dim];
        std::vector<std::size_t> out_dims;
        for (int i = 0; i < r; ++i)
        {
            if (i == dim)
            {
                if (keepdim)
                    out_dims.push_back(1);
            }
            else
            {
                out_dims.push_back(in_dims[i]);
            }
        }
        if (out_dims.empty())
            out_dims.push_back(1);
        Size outer = 1;
        for (int i = 0; i < dim; ++i)
            outer *= in_dims[i];
        Size inner = 1;
        for (int i = dim + 1; i < r; ++i)
            inner *= in_dims[i];
        Shape out_shape(out_dims);
        Tensor out = Tensor::zeros(out_shape, false);
        Tensor::value_type *odata = out.data();
        const Tensor::value_type *idata = a.data();
        for (Size o = 0; o < outer; ++o)
        {
            for (Size i = 0; i < inner; ++i)
            {
                Tensor::value_type best = idata[o * dim_size * inner + i];
                Size best_idx = 0;
                for (Size d = 1; d < dim_size; ++d)
                {
                    Tensor::value_type v = idata[o * dim_size * inner + d * inner + i];
                    if (v < best)
                    {
                        best = v;
                        best_idx = d;
                    }
                }
                odata[o * inner + i] = static_cast<Tensor::value_type>(best_idx);
            }
        }
        return out;
    }

} // namespace synara