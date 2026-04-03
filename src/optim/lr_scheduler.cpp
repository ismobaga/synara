#include "synara/optim/lr_scheduler.hpp"

#include <cmath>
#include <limits>

#include "synara/core/error.hpp"

namespace synara
{

    LRScheduler::LRScheduler(Optimizer &optimizer)
        : optimizer_(optimizer), step_count_(0)
    {
    }

    void LRScheduler::step()
    {
        ++step_count_;
    }

    double LRScheduler::get_lr() const noexcept
    {
        return optimizer_.learning_rate();
    }

    StepLR::StepLR(Optimizer &optimizer, std::size_t step_size, double gamma)
        : LRScheduler(optimizer), step_size_(step_size), gamma_(gamma)
    {
        if (step_size_ == 0)
            throw ValueError("StepLR: step_size must be > 0.");
        if (gamma_ <= 0.0)
            throw ValueError("StepLR: gamma must be > 0.");
    }

    void StepLR::step()
    {
        ++step_count_;
        if (step_count_ % step_size_ == 0)
        {
            optimizer_.set_learning_rate(optimizer_.learning_rate() * gamma_);
        }
    }

    ExponentialLR::ExponentialLR(Optimizer &optimizer, double gamma)
        : LRScheduler(optimizer), gamma_(gamma)
    {
        if (gamma_ <= 0.0)
            throw ValueError("ExponentialLR: gamma must be > 0.");
    }

    void ExponentialLR::step()
    {
        ++step_count_;
        optimizer_.set_learning_rate(optimizer_.learning_rate() * gamma_);
    }

    CosineAnnealingLR::CosineAnnealingLR(Optimizer &optimizer, std::size_t t_max, double eta_min)
        : LRScheduler(optimizer), t_max_(t_max), eta_min_(eta_min), initial_lr_(optimizer.learning_rate())
    {
        if (t_max_ == 0)
            throw ValueError("CosineAnnealingLR: T_max must be > 0.");
    }

    void CosineAnnealingLR::step()
    {
        ++step_count_;
        const double t_cur = static_cast<double>(step_count_ % t_max_);
        const double cosine = std::cos(3.14159265358979323846 * (t_cur / static_cast<double>(t_max_)));
        const double lr = eta_min_ + 0.5 * (initial_lr_ - eta_min_) * (1.0 + cosine);
        optimizer_.set_learning_rate(lr);
    }

    ReduceLROnPlateau::ReduceLROnPlateau(Optimizer &optimizer, double factor, std::size_t patience)
        : LRScheduler(optimizer), factor_(factor), patience_(patience), bad_epochs_(0),
          best_(std::numeric_limits<double>::infinity()), has_best_(false)
    {
        if (factor_ <= 0.0 || factor_ >= 1.0)
            throw ValueError("ReduceLROnPlateau: factor must be in (0, 1).");
    }

    void ReduceLROnPlateau::step()
    {
        ++step_count_;
    }

    void ReduceLROnPlateau::step(double metric)
    {
        ++step_count_;
        if (!has_best_ || metric < best_)
        {
            best_ = metric;
            has_best_ = true;
            bad_epochs_ = 0;
            return;
        }

        ++bad_epochs_;
        if (bad_epochs_ > patience_)
        {
            optimizer_.set_learning_rate(optimizer_.learning_rate() * factor_);
            bad_epochs_ = 0;
        }
    }

} // namespace synara
