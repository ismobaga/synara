#pragma once

#include <string>

#include "synara/nn/module.hpp"

namespace synara
{

    // Text-based checkpoint (human-readable, legacy format)
    void save_state_dict(const StateDict &state, const std::string &path);
    StateDict load_state_dict(const std::string &path);

    // Binary checkpoint format (compact, efficient)
    void save_state_dict_binary(const StateDict &state, const std::string &path);
    StateDict load_state_dict_binary(const std::string &path);

} // namespace synara
