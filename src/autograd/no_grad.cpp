#include "synara/autograd/no_grad.hpp"

namespace synara
{
    thread_local bool g_grad_mode = true;

    bool grad_mode_enabled()
    {
        return g_grad_mode;
    }

    void set_grad_mode(bool enabled)
    {
        g_grad_mode = enabled;
    }

    NoGradGuard::NoGradGuard() : prev_(g_grad_mode)
    {
        g_grad_mode = false;
    }

    NoGradGuard::~NoGradGuard()
    {
        g_grad_mode = prev_;
    }

    NoGradGuard no_grad()
    {
        return NoGradGuard{};
    }

} // namespace synara
