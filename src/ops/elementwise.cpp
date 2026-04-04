#include "synara/ops/elementwise.hpp"
#include "synara/autograd/nodes.hpp"

#include <cmath>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#include "synara/core/error.hpp"

namespace synara
{
    namespace
    {

        enum class BinaryOpKind
        {
            Add,
            Sub,
            Mul,
            Div,
        };

        enum class ScalarOpKind
        {
            Add,
            Sub,
            Mul,
            Div,
            ReverseSub,
            ReverseDiv,
        };

        static void require_same_shape(const Tensor &a, const Tensor &b, const char *op_name)
        {
            if (a.shape() != b.shape())
            {
                throw ShapeError(std::string(op_name) + ": tensors must have the same shape.");
            }
        }

        static void require_contiguous(const Tensor &t, const char *op_name)
        {
            if (!t.is_contiguous())
            {
                throw ShapeError(std::string(op_name) + ": tensors must be contiguous.");
            }
        }

        static Tensor::value_type apply_binary_scalar(BinaryOpKind kind, Tensor::value_type x, Tensor::value_type y)
        {
            switch (kind)
            {
            case BinaryOpKind::Add:
                return x + y;
            case BinaryOpKind::Sub:
                return x - y;
            case BinaryOpKind::Mul:
                return x * y;
            case BinaryOpKind::Div:
                return x / y;
            }
            return x;
        }

        static Tensor::value_type apply_scalar_scalar(ScalarOpKind kind, Tensor::value_type x, Tensor::value_type scalar)
        {
            switch (kind)
            {
            case ScalarOpKind::Add:
                return x + scalar;
            case ScalarOpKind::Sub:
                return x - scalar;
            case ScalarOpKind::Mul:
                return x * scalar;
            case ScalarOpKind::Div:
                return x / scalar;
            case ScalarOpKind::ReverseSub:
                return scalar - x;
            case ScalarOpKind::ReverseDiv:
                return scalar / x;
            }
            return x;
        }

        bool compute_requires_grad(const Tensor &a, const Tensor &b)
        {
            return a.requires_grad() || b.requires_grad();
        }

        bool should_parallelize_elementwise(Size numel)
        {
            return static_cast<long long>(numel) >= (1LL << 15);
        }

        static Tensor unary_scalar_op(const Tensor &a, Tensor::value_type scalar, ScalarOpKind kind)
        {
            require_contiguous(a, "unary scalar op");
            Tensor out = Tensor::zeros(a.shape());

            const Tensor::value_type *in = a.data();
            Tensor::value_type *out_data = out.data();
            const Size numel = a.numel();
            const bool parallel = should_parallelize_elementwise(numel);

#if defined(__SSE2__)
            const Size pair_count = numel / 2;
            const __m128d scalar_vec = _mm_set1_pd(scalar);
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long pair = 0; pair < static_cast<long long>(pair_count); ++pair)
            {
                const Size i = static_cast<Size>(pair) * 2;
                const __m128d x = _mm_loadu_pd(in + i);
                __m128d y;

                switch (kind)
                {
                case ScalarOpKind::Add:
                    y = _mm_add_pd(x, scalar_vec);
                    break;
                case ScalarOpKind::Sub:
                    y = _mm_sub_pd(x, scalar_vec);
                    break;
                case ScalarOpKind::Mul:
                    y = _mm_mul_pd(x, scalar_vec);
                    break;
                case ScalarOpKind::Div:
                    y = _mm_div_pd(x, scalar_vec);
                    break;
                case ScalarOpKind::ReverseSub:
                    y = _mm_sub_pd(scalar_vec, x);
                    break;
                case ScalarOpKind::ReverseDiv:
                    y = _mm_div_pd(scalar_vec, x);
                    break;
                }

                _mm_storeu_pd(out_data + i, y);
            }

            for (Size i = pair_count * 2; i < numel; ++i)
            {
                out_data[i] = apply_scalar_scalar(kind, in[i], scalar);
            }
#else
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(numel); ++i)
            {
                out_data[static_cast<Size>(i)] = apply_scalar_scalar(kind, in[static_cast<Size>(i)], scalar);
            }
#endif

            return out;
        }

        static Tensor binary_tensor_op(const Tensor &a, const Tensor &b, BinaryOpKind kind)
        {
            require_same_shape(a, b, "elementwise op");
            require_contiguous(a, "elementwise op");
            require_contiguous(b, "elementwise op");

            Tensor out = Tensor::zeros(a.shape());
            const Tensor::value_type *a_data = a.data();
            const Tensor::value_type *b_data = b.data();
            Tensor::value_type *out_data = out.data();
            const Size numel = a.numel();
            const bool parallel = should_parallelize_elementwise(numel);

#if defined(__SSE2__)
            const Size pair_count = numel / 2;
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long pair = 0; pair < static_cast<long long>(pair_count); ++pair)
            {
                const Size i = static_cast<Size>(pair) * 2;
                const __m128d x = _mm_loadu_pd(a_data + i);
                const __m128d y = _mm_loadu_pd(b_data + i);
                __m128d z;

                switch (kind)
                {
                case BinaryOpKind::Add:
                    z = _mm_add_pd(x, y);
                    break;
                case BinaryOpKind::Sub:
                    z = _mm_sub_pd(x, y);
                    break;
                case BinaryOpKind::Mul:
                    z = _mm_mul_pd(x, y);
                    break;
                case BinaryOpKind::Div:
                    z = _mm_div_pd(x, y);
                    break;
                }

                _mm_storeu_pd(out_data + i, z);
            }

            for (Size i = pair_count * 2; i < numel; ++i)
            {
                out_data[i] = apply_binary_scalar(kind, a_data[i], b_data[i]);
            }
#else
#if defined(SYNARA_USE_OPENMP)
#pragma omp parallel for if (parallel) schedule(static)
#endif
            for (long long i = 0; i < static_cast<long long>(numel); ++i)
            {
                out_data[static_cast<Size>(i)] =
                    apply_binary_scalar(kind, a_data[static_cast<Size>(i)], b_data[static_cast<Size>(i)]);
            }
#endif

            return out;
        }

    } // namespace

    Tensor add(const Tensor &a, const Tensor &b)
    {
        require_same_shape(a, b, "add");
        Tensor out = binary_tensor_op(a, b, BinaryOpKind::Add);
        bool req = compute_requires_grad(a, b);
        out.set_leaf(!req);
        out.set_requires_grad(req);
        if (req)
        {
            auto node = std::make_shared<AddNode>(a, b);
            out.set_grad_fn(node);
        }
        return out;
    }

    Tensor sub(const Tensor &a, const Tensor &b)
    {
        require_same_shape(a, b, "sub");
        Tensor out = binary_tensor_op(a, b, BinaryOpKind::Sub);
        bool req = compute_requires_grad(a, b);
        out.set_leaf(!req);
        out.set_requires_grad(req);
        if (req)
        {
            auto node = std::make_shared<SubNode>(a, b);
            out.set_grad_fn(node);
        }
        return out;
    }

    Tensor mul(const Tensor &a, const Tensor &b)
    {
        require_same_shape(a, b, "mul");
        Tensor out = binary_tensor_op(a, b, BinaryOpKind::Mul);
        bool req = compute_requires_grad(a, b);
        out.set_leaf(!req);
        out.set_requires_grad(req);

        if (req)
        {
            auto node = std::make_shared<MulNode>(a, b);
            out.set_grad_fn(node);
        }
        return out;
    }

    Tensor div(const Tensor &a, const Tensor &b)
    {
        require_same_shape(a, b, "div");
        Tensor out = binary_tensor_op(a, b, BinaryOpKind::Div);
        bool req = compute_requires_grad(a, b);
        out.set_leaf(!req);
        out.set_requires_grad(req);
        if (req)
        {
            auto node = std::make_shared<DivNode>(a, b);
            out.set_grad_fn(node);
        }
        return out;
    }

    Tensor add(const Tensor &a, Tensor::value_type scalar)
    {
        return unary_scalar_op(a, scalar, ScalarOpKind::Add);
    }

    Tensor sub(const Tensor &a, Tensor::value_type scalar)
    {
        return unary_scalar_op(a, scalar, ScalarOpKind::Sub);
    }

    Tensor mul(const Tensor &a, Tensor::value_type scalar)
    {
        return unary_scalar_op(a, scalar, ScalarOpKind::Mul);
    }

    Tensor div(const Tensor &a, Tensor::value_type scalar)
    {
        return unary_scalar_op(a, scalar, ScalarOpKind::Div);
    }

    Tensor add(Tensor::value_type scalar, const Tensor &a)
    {
        return add(a, scalar);
    }

    Tensor sub(Tensor::value_type scalar, const Tensor &a)
    {
        return unary_scalar_op(a, scalar, ScalarOpKind::ReverseSub);
    }

    Tensor mul(Tensor::value_type scalar, const Tensor &a)
    {
        return mul(a, scalar);
    }

    Tensor div(Tensor::value_type scalar, const Tensor &a)
    {
        return unary_scalar_op(a, scalar, ScalarOpKind::ReverseDiv);
    }

} // namespace synara