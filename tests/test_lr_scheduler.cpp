#include <cassert>
#include <cmath>

#include "synara/optim/lr_scheduler.hpp"
#include "synara/optim/sgd.hpp"

int main()
{
    using namespace synara;

    Tensor p(Shape({1}), true);
    std::vector<Tensor *> params = {&p};

    SGD opt_step(params, 0.1);
    StepLR step_lr(opt_step, 2, 0.5);
    step_lr.step();
    assert(std::fabs(opt_step.learning_rate() - 0.1) < 1e-12);
    step_lr.step();
    assert(std::fabs(opt_step.learning_rate() - 0.05) < 1e-12);

    SGD opt_exp(params, 0.2);
    ExponentialLR exp_lr(opt_exp, 0.5);
    exp_lr.step();
    assert(std::fabs(opt_exp.learning_rate() - 0.1) < 1e-12);

    SGD opt_cos(params, 1.0);
    CosineAnnealingLR cos_lr(opt_cos, 4, 0.0);
    cos_lr.step();
    assert(opt_cos.learning_rate() < 1.0 && opt_cos.learning_rate() > 0.0);

    SGD opt_plateau(params, 0.3);
    ReduceLROnPlateau plateau(opt_plateau, 0.5, 1);
    plateau.step(1.0);
    plateau.step(1.1);
    plateau.step(1.2);
    assert(std::fabs(opt_plateau.learning_rate() - 0.15) < 1e-12);

    return 0;
}
