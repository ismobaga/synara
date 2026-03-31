#include <cassert>
#include "synara/core/types.hpp"
#include "synara/tensor/shape.hpp"

int main() {
    using namespace synara;

    Shape s({2, 3, 4});
    assert(s.rank() == 3);
    assert(s.numel() == 24);
    assert(s[0] == 2);
    assert(s[1] == 3);
    assert(s[2] == 4);

    Shape scalar({});
    assert(scalar.rank() == 0);
    assert(scalar.numel() == 1);

    return 0;
}