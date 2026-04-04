#include <cassert>

#include "synara/data/dataloader.hpp"
#include "synara/data/dataset.hpp"
#include "synara/nn/linear.hpp"
#include "synara/ops/loss.hpp"
#include "synara/optim/sgd.hpp"
#include "synara/train/trainer.hpp"

int main()
{
    using namespace synara;

    Tensor x = Tensor::from_vector(
        Shape({4, 1}),
        {0.0f, 1.0f, 2.0f, 3.0f},
        false);
    Tensor y = Tensor::from_vector(
        Shape({4, 1}),
        {0.0f, 2.0f, 4.0f, 6.0f},
        false);

    TensorDataset dataset(x, y);
    DataLoader loader(dataset, 2, false, 42);

    Linear model(1, 1, true);
    std::vector<Tensor *> params;
    for (auto *p : model.parameters())
    {
        params.push_back(&p->tensor());
    }
    SGD optimizer(params, 0.01);

    auto loss_fn = [](const Tensor &pred, const Tensor &target)
    {
        return mse_loss(pred, target);
    };

    reset_profile_data();
    enable_profiling(true);

    const EpochStats train_stats = train_epoch_profiled(model, loader, optimizer, loss_fn);
    const Tensor::value_type train_loss = train_epoch(model, loader, optimizer, loss_fn);
    const EpochStats eval_stats = eval_epoch_profiled(model, loader, loss_fn);
    const Tensor::value_type eval_loss = eval_epoch(model, loader, loss_fn);

    assert(train_stats.batches == 2);
    assert(train_stats.mean_loss >= 0.0f);
    assert(train_stats.total_ms >= 0.0);
    assert(train_stats.forward_ms >= 0.0);
    assert(train_stats.backward_ms >= 0.0);
    assert(train_stats.step_ms >= 0.0);
    assert(train_stats.average_batch_ms() >= 0.0);

    assert(eval_stats.batches == 2);
    assert(eval_stats.mean_loss >= 0.0f);
    assert(eval_stats.total_ms >= 0.0);
    assert(eval_stats.forward_ms >= 0.0);
    assert(eval_stats.backward_ms == 0.0);
    assert(eval_stats.step_ms == 0.0);

    assert(get_profile_stats("train_epoch.total").calls >= 1);
    assert(get_profile_stats("train_epoch.forward").calls >= train_stats.batches);
    assert(get_profile_stats("train_epoch.backward").calls >= train_stats.batches);
    assert(get_profile_stats("train_epoch.step").calls >= train_stats.batches);
    assert(get_profile_stats("eval_epoch.total").calls >= 1);
    assert(get_profile_stats("eval_epoch.forward").calls >= eval_stats.batches);

    assert(train_loss >= 0.0f);
    assert(eval_loss >= 0.0f);

    return 0;
}
