#pragma once

#include <string>

#include "synara/nn/module.hpp"

namespace synara
{

    void save_state_dict(const StateDict &state, const std::string &path);
    StateDict load_state_dict(const std::string &path);

} // namespace synara
