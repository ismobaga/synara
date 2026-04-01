#pragma once

#include <cstdint>

#include "synara/nn/module.hpp"

namespace synara
{

    class Dropout : public Module
    {
    public:
        explicit Dropout(Tensor::value_type p = 0.5f, std::uint64_t seed = 0xC0FFEEULL);

        Tensor forward(const Tensor &input) override;
        std::vector<Parameter *> parameters() override { return {}; }

        void train() noexcept;
        void eval() noexcept;
        bool is_training() const noexcept;

        Tensor::value_type p() const noexcept;
        void set_p(Tensor::value_type p);

        std::uint64_t seed() const noexcept;
        void set_seed(std::uint64_t seed) noexcept;

    private:
        Tensor::value_type next_uniform01();

        Tensor::value_type p_;
        bool training_;
        std::uint64_t rng_state_;
    };

} // namespace synara
