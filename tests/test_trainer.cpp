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

    const Tensor::value_type train_loss = train_epoch(model, loader, optimizer, loss_fn);
    const Tensor::value_type eval_loss = eval_epoch(model, loader, loss_fn);

    assert(train_loss >= 0.0f);
    assert(eval_loss >= 0.0f);

    return 0;
}
