#include "synara/ops/linalg.hpp"
#include "synara/autograd/nodes.hpp"
#include "synara/core/error.hpp"
#include <stdexcept>
#include <string>

namespace synara
{

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

        Tensor out = Tensor::zeros(Shape({m, n}));

        for (Size i = 0; i < m; ++i)
        {
            for (Size j = 0; j < n; ++j)
            {
                Tensor::value_type acc = 0;
                for (Size k = 0; k < k1; ++k)
                {
                    acc += a.at({i, k}) * b.at({k, j});
                }
                out.at({i, j}) = acc;
            }
        }

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
            for (Size d = 0; d < embed_dim; ++d)
                o[i * embed_dim + d] = w[idx * embed_dim + d];
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
