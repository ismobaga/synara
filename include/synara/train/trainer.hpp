#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include "synara/autograd/no_grad.hpp"
#include "synara/core/profiler.hpp"
#include "synara/data/dataloader.hpp"
#include "synara/nn/module.hpp"
#include "synara/optim/optimizer.hpp"

namespace synara
{

    struct EpochStats
    {
        Tensor::value_type mean_loss = 0.0f;
        Size batches = 0;
        double total_ms = 0.0;
        double data_ms = 0.0;
        double forward_ms = 0.0;
        double loss_ms = 0.0;
        double zero_grad_ms = 0.0;
        double backward_ms = 0.0;
        double step_ms = 0.0;

        double average_batch_ms() const noexcept
        {
            return batches == 0 ? 0.0 : total_ms / static_cast<double>(batches);
        }
    };

    inline std::string format_epoch_stats_csv(const EpochStats &stats, bool include_header = true)
    {
        std::ostringstream oss;
        if (include_header)
        {
            oss << "mean_loss,batches,total_ms,data_ms,forward_ms,loss_ms,zero_grad_ms,backward_ms,step_ms,avg_batch_ms\n";
        }

        oss << std::fixed << std::setprecision(3)
            << static_cast<double>(stats.mean_loss) << ","
            << stats.batches << ","
            << stats.total_ms << ","
            << stats.data_ms << ","
            << stats.forward_ms << ","
            << stats.loss_ms << ","
            << stats.zero_grad_ms << ","
            << stats.backward_ms << ","
            << stats.step_ms << ","
            << stats.average_batch_ms();
        return oss.str();
    }

    inline std::string format_epoch_stats_json(const EpochStats &stats)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3)
            << "{"
            << "\"mean_loss\":" << static_cast<double>(stats.mean_loss)
            << ",\"batches\":" << stats.batches
            << ",\"total_ms\":" << stats.total_ms
            << ",\"data_ms\":" << stats.data_ms
            << ",\"forward_ms\":" << stats.forward_ms
            << ",\"loss_ms\":" << stats.loss_ms
            << ",\"zero_grad_ms\":" << stats.zero_grad_ms
            << ",\"backward_ms\":" << stats.backward_ms
            << ",\"step_ms\":" << stats.step_ms
            << ",\"avg_batch_ms\":" << stats.average_batch_ms()
            << "}";
        return oss.str();
    }

    inline bool write_epoch_stats_csv(const EpochStats &stats, const std::string &path, bool include_header = true)
    {
        std::ofstream out(path);
        if (!out)
        {
            return false;
        }

        out << format_epoch_stats_csv(stats, include_header);
        return static_cast<bool>(out);
    }

    inline bool write_epoch_stats_json(const EpochStats &stats, const std::string &path)
    {
        std::ofstream out(path);
        if (!out)
        {
            return false;
        }

        out << format_epoch_stats_json(stats);
        return static_cast<bool>(out);
    }

    namespace detail
    {
        template <typename Fn>
        double measure_ms(Fn &&fn)
        {
            const auto start = std::chrono::steady_clock::now();
            std::forward<Fn>(fn)();
            const auto end = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(end - start).count();
        }
    } // namespace detail

    template <typename LossFn>
    EpochStats train_epoch_profiled(
        Module &model,
        DataLoader &loader,
        Optimizer &optimizer,
        LossFn loss_fn,
        const std::string &scope_prefix = "train_epoch")
    {
        model.train();

        const std::string total_scope_name = scope_prefix + ".total";
        const std::string batch_scope_name = scope_prefix + ".batch";
        const std::string data_scope_name = scope_prefix + ".data";
        const std::string forward_scope_name = scope_prefix + ".forward";
        const std::string loss_scope_name = scope_prefix + ".loss";
        const std::string zero_grad_scope_name = scope_prefix + ".zero_grad";
        const std::string backward_scope_name = scope_prefix + ".backward";
        const std::string step_scope_name = scope_prefix + ".step";

        EpochStats stats;
        Tensor::value_type loss_sum = 0.0f;
        const auto epoch_start = std::chrono::steady_clock::now();
        ScopedProfile total_scope(total_scope_name);

        for (auto it = loader.begin(); it != loader.end(); ++it)
        {
            const auto batch_start = std::chrono::steady_clock::now();
            ScopedProfile batch_scope(batch_scope_name);

            std::pair<Tensor, Tensor> batch;
            stats.data_ms += detail::measure_ms([&]()
                                                {
                ScopedProfile scope(data_scope_name);
                batch = *it; });

            Tensor predictions;
            stats.forward_ms += detail::measure_ms([&]()
                                                   {
                ScopedProfile scope(forward_scope_name);
                predictions = model(batch.first); });

            Tensor loss;
            stats.loss_ms += detail::measure_ms([&]()
                                                {
                ScopedProfile scope(loss_scope_name);
                loss = loss_fn(predictions, batch.second); });

            stats.zero_grad_ms += detail::measure_ms([&]()
                                                     {
                ScopedProfile scope(zero_grad_scope_name);
                optimizer.zero_grad(); });

            stats.backward_ms += detail::measure_ms([&]()
                                                    {
                ScopedProfile scope(backward_scope_name);
                loss.backward(); });

            stats.step_ms += detail::measure_ms([&]()
                                                {
                ScopedProfile scope(step_scope_name);
                optimizer.step(); });

            loss_sum += loss.item();
            ++stats.batches;
            stats.total_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - batch_start)
                                  .count();
        }

        stats.total_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - epoch_start)
                             .count();
        stats.mean_loss = stats.batches == 0
                              ? 0.0f
                              : loss_sum / static_cast<Tensor::value_type>(stats.batches);
        return stats;
    }

    template <typename LossFn>
    Tensor::value_type train_epoch(Module &model, DataLoader &loader, Optimizer &optimizer, LossFn loss_fn)
    {
        return train_epoch_profiled(model, loader, optimizer, std::move(loss_fn)).mean_loss;
    }

    template <typename LossFn>
    EpochStats eval_epoch_profiled(
        Module &model,
        DataLoader &loader,
        LossFn loss_fn,
        const std::string &scope_prefix = "eval_epoch")
    {
        model.eval();
        NoGradGuard guard = no_grad();

        const std::string total_scope_name = scope_prefix + ".total";
        const std::string batch_scope_name = scope_prefix + ".batch";
        const std::string data_scope_name = scope_prefix + ".data";
        const std::string forward_scope_name = scope_prefix + ".forward";
        const std::string loss_scope_name = scope_prefix + ".loss";

        EpochStats stats;
        Tensor::value_type loss_sum = 0.0f;
        const auto epoch_start = std::chrono::steady_clock::now();
        ScopedProfile total_scope(total_scope_name);

        for (auto it = loader.begin(); it != loader.end(); ++it)
        {
            const auto batch_start = std::chrono::steady_clock::now();
            ScopedProfile batch_scope(batch_scope_name);

            std::pair<Tensor, Tensor> batch;
            stats.data_ms += detail::measure_ms([&]()
                                                {
                ScopedProfile scope(data_scope_name);
                batch = *it; });

            Tensor predictions;
            stats.forward_ms += detail::measure_ms([&]()
                                                   {
                ScopedProfile scope(forward_scope_name);
                predictions = model(batch.first); });

            Tensor loss;
            stats.loss_ms += detail::measure_ms([&]()
                                                {
                ScopedProfile scope(loss_scope_name);
                loss = loss_fn(predictions, batch.second); });

            loss_sum += loss.item();
            ++stats.batches;
            stats.total_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - batch_start)
                                  .count();
        }

        stats.total_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - epoch_start)
                             .count();
        stats.mean_loss = stats.batches == 0
                              ? 0.0f
                              : loss_sum / static_cast<Tensor::value_type>(stats.batches);
        return stats;
    }

    template <typename LossFn>
    Tensor::value_type eval_epoch(Module &model, DataLoader &loader, LossFn loss_fn)
    {
        return eval_epoch_profiled(model, loader, std::move(loss_fn)).mean_loss;
    }

} // namespace synara
