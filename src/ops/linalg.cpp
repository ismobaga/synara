#include "synara/ops/linalg.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/core/error.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace synara
{
    namespace
    {
        bool should_parallelize_matmul(Size m, Size k, Size n)
        {
            return static_cast<long long>(m) *
                       static_cast<long long>(k) *
                       static_cast<long long>(n) >=
                   (1LL << 15);
        }

        Tensor matmul_contiguous(const Tensor &a, const Tensor &b, Size m, Size k, Size n)
        {
            Tensor out = Tensor::zeros(Shape({m, n}));

            const Tensor::value_type *a_data = a.data();
            const Tensor::value_type *b_data = b.data();
            Tensor::value_type *out_data = out.data();
            const bool parallel = should_parallelize_matmul(m, k, n);

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(m); ++i)
            {
                const Size row = static_cast<Size>(i);
                const Tensor::value_type *a_row = a_data + row * k;
                Tensor::value_type *out_row = out_data + row * n;

                for (Size kk = 0; kk < k; ++kk)
                {
                    const Tensor::value_type a_ik = a_row[kk];
                    const Tensor::value_type *b_row = b_data + kk * n;

                    for (Size j = 0; j < n; ++j)
                    {
                        out_row[j] += a_ik * b_row[j];
                    }
                }
            }

            return out;
        }

        Tensor matmul_strided(const Tensor &a, const Tensor &b, Size m, Size k, Size n)
        {
            Tensor out = Tensor::zeros(Shape({m, n}));

            const Tensor::value_type *a_data = a.data();
            const Tensor::value_type *b_data = b.data();
            Tensor::value_type *out_data = out.data();

            const auto a_strides = a.strides().values();
            const auto b_strides = b.strides().values();
            const Size a_row_stride = a_strides[0];
            const Size a_col_stride = a_strides[1];
            const Size b_row_stride = b_strides[0];
            const Size b_col_stride = b_strides[1];
            const bool parallel = should_parallelize_matmul(m, k, n);

#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(m); ++i)
            {
                const Size row = static_cast<Size>(i);
                const Size a_row_base = row * a_row_stride;
                Tensor::value_type *out_row = out_data + row * n;

                for (Size j = 0; j < n; ++j)
                {
                    Tensor::value_type acc = 0.0;
                    const Size b_col_base = j * b_col_stride;

                    for (Size kk = 0; kk < k; ++kk)
                    {
                        acc += a_data[a_row_base + kk * a_col_stride] *
                               b_data[kk * b_row_stride + b_col_base];
                    }

                    out_row[j] = acc;
                }
            }

            return out;
        }
    } // namespace

    Tensor matmul(const Tensor &a, const Tensor &b)
    {
        if (a.rank() != 2 || b.rank() != 2)
        {
            throw ShapeError("matmul(): only rank-2 tensors are supported.");
        }

        const Size m = a.shape()[0];
        const Size k1 = a.shape()[1];
        const Size k2 = b.shape()[0];
        const Size n = b.shape()[1];

        if (k1 != k2)
        {
            throw ShapeError("matmul(): inner dimensions must match.");
        }

        Tensor out = (a.is_contiguous() && b.is_contiguous())
                         ? matmul_contiguous(a, b, m, k1, n)
                         : matmul_strided(a, b, m, k1, n);

        bool req = a.requires_grad() || b.requires_grad();
        out.set_leaf(!req);
        out.set_requires_grad(req);
        if (req)
        {
            auto node = std::make_shared<MatMulNode>(a, b);
            out.set_grad_fn(node);
        }

        return out;
    }

    Tensor embedding(const Tensor &indices, const Tensor &weight)
    {
        if (indices.rank() != 1)
            throw ShapeError("embedding(): indices must be a rank-1 tensor.");
        if (weight.rank() != 2)
            throw ShapeError("embedding(): weight must be rank-2 (vocabulary_size x embedding_dim).");

        const Size n = indices.numel();
        const Size vocab_size = weight.shape()[0];
        const Size embed_dim = weight.shape()[1];

        Tensor out = Tensor::zeros(Shape({n, embed_dim}));
        const Tensor::value_type *w = weight.data();
        Tensor::value_type *o = out.data();

        for (Size i = 0; i < n; ++i)
        {
            const auto idx = static_cast<Size>(indices.data()[i]);
            if (idx >= vocab_size)
                throw std::out_of_range("embedding(): index " + std::to_string(idx) +
                                        " out of range for vocab size " +
                                        std::to_string(vocab_size));

            std::copy_n(w + idx * embed_dim, embed_dim, o + i * embed_dim);
        }

        bool req = weight.requires_grad() && grad_mode_enabled();
        out.set_leaf(!req);
        out.set_requires_grad(req);
        if (req)
        {
            auto node = std::make_shared<EmbeddingNode>(indices, weight);
            out.set_grad_fn(node);
        }
        return out;
    }

}
