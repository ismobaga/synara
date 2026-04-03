#pragma once

#include "synara/tensor/tensor.hpp"

namespace synara
{

    // Type aliases for different precision tensors
    // Currently, Tensor is hardcoded to double precision
    // In the future, we can template Tensor<T> to support both float32 and float64

    // Float32 support through type erasure and storage
    // For now, we provide utilities to work with float32 data

    // Factory functions for creating float32 tensors
    // Note: These convert to double internally for compatibility
    template <typename Iter>
    inline Tensor make_float32_tensor(const Shape &shape, Iter begin, Iter end)
    {
        std::vector<double> values;
        for (auto it = begin; it != end; ++it)
        {
            values.push_back(static_cast<double>(*it));
        }
        return Tensor::from_vector(shape, values, false);
    }

    template <typename Container>
    inline Tensor make_float32_tensor(const Shape &shape, const Container &data)
    {
        return make_float32_tensor(shape, data.begin(), data.end());
    }

    // Float32 data extraction utilities
    inline std::vector<float> to_float32(const Tensor &t)
    {
        std::vector<float> result;
        const double *data = t.data();
        for (std::size_t i = 0; i < t.numel(); ++i)
        {
            result.push_back(static_cast<float>(data[i]));
        }
        return result;
    }

    // Type information utility
    enum class DataType
    {
        FLOAT32,
        FLOAT64,
        UNKNOWN
    };

    inline DataType get_tensor_dtype(const Tensor &t)
    {
        // Currently all tensors are float64 internally
        return DataType::FLOAT64;
    }

    // Dtype conversion/casting utilities (primarily for future use)
    inline std::string dtype_to_string(DataType dtype)
    {
        switch (dtype)
        {
        case DataType::FLOAT32:
            return "float32";
        case DataType::FLOAT64:
            return "float64";
        default:
            return "unknown";
        }
    }

} // namespace synara
