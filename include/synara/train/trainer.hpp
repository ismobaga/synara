#pragma once

#include <utility>

#include "synara/autograd/no_grad.hpp"
#include "synara/data/dataloader.hpp"
#include "synara/nn/module.hpp"
#include "synara/optim/optimizer.hpp"

namespace synara
{

    template <typename LossFn>
    Tensor::value_type train_epoch(Module &model, DataLoader &loader, Optimizer &optimizer, LossFn loss_fn)
    {
        model.train();

        Tensor::value_type loss_sum = 0.0f;
        Size batches = 0;
        for (auto it = loader.begin(); it != loader.end(); ++it)
        {
            auto batch = *it;
            Tensor predictions = model(batch.first);
            Tensor loss = loss_fn(predictions, batch.second);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            loss_sum += loss.item();
            ++batches;
        }

        return batches == 0 ? 0.0f : loss_sum / static_cast<Tensor::value_type>(batches);
    }

    template <typename LossFn>
    Tensor::value_type eval_epoch(Module &model, DataLoader &loader, LossFn loss_fn)
    {
        model.eval();
        NoGradGuard guard = no_grad();

        Tensor::value_type loss_sum = 0.0f;
        Size batches = 0;
        for (auto it = loader.begin(); it != loader.end(); ++it)
        {
            auto batch = *it;
            Tensor predictions = model(batch.first);
            Tensor loss = loss_fn(predictions, batch.second);
            loss_sum += loss.item();
            ++batches;
        }

        return batches == 0 ? 0.0f : loss_sum / static_cast<Tensor::value_type>(batches);
    }

} // namespace synara
