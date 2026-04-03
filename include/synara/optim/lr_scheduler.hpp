#pragma once

#include <cstddef>

#include "synara/optim/optimizer.hpp"

namespace synara
{

    class LRScheduler
    {
    public:
        explicit LRScheduler(Optimizer &optimizer);
        virtual ~LRScheduler() = default;

        virtual void step();
        double get_lr() const noexcept;

    protected:
        Optimizer &optimizer_;
        std::size_t step_count_;
    };

    class StepLR : public LRScheduler
    {
    public:
        StepLR(Optimizer &optimizer, std::size_t step_size, double gamma);
        void step() override;

    private:
        std::size_t step_size_;
        double gamma_;
    };

    class ExponentialLR : public LRScheduler
    {
    public:
        ExponentialLR(Optimizer &optimizer, double gamma);
        void step() override;

    private:
        double gamma_;
    };

    class CosineAnnealingLR : public LRScheduler
    {
    public:
        CosineAnnealingLR(Optimizer &optimizer, std::size_t t_max, double eta_min = 0.0);
        void step() override;

    private:
        std::size_t t_max_;
        double eta_min_;
        double initial_lr_;
    };

    class ReduceLROnPlateau : public LRScheduler
    {
    public:
        ReduceLROnPlateau(Optimizer &optimizer, double factor = 0.1, std::size_t patience = 10);
        void step() override;
        void step(double metric);

    private:
        double factor_;
        std::size_t patience_;
        std::size_t bad_epochs_;
        double best_;
        bool has_best_;
    };

} // namespace synara
