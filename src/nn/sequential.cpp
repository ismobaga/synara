#include "synara/nn/sequential.hpp"

#include <utility>

namespace synara
{

    Sequential::Sequential(std::vector<std::shared_ptr<Module>> modules)
        : modules_(std::move(modules)) {}

    void Sequential::add(std::shared_ptr<Module> module)
    {
        modules_.push_back(std::move(module));
    }

    Tensor Sequential::forward(const Tensor &input)
    {
        Tensor output = input;
        for (const std::shared_ptr<Module> &module : modules_)
        {
            output = (*module)(output);
        }
        return output;
    }

    std::vector<Parameter *> Sequential::parameters()
    {
        std::vector<Parameter *> out;
        for (const std::shared_ptr<Module> &module : modules_)
        {
            std::vector<Parameter *> child_params = module->parameters();
            out.insert(out.end(), child_params.begin(), child_params.end());
        }
        return out;
    }

    Size Sequential::size() const noexcept
    {
        return modules_.size();
    }

} // namespace synara
