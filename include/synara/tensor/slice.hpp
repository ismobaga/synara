#pragma once

#include <optional>

namespace synara
{

    struct Slice
    {
        std::optional<long long> start;
        std::optional<long long> stop;
        long long step = 1;

        static Slice all() { return Slice{}; }
    };

} // namespace synara
