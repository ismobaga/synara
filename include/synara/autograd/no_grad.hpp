#pragma once
namespace synara
{
    bool grad_mode_enabled();
    void set_grad_mode(bool enabled);
    struct NoGradGuard
    {
        NoGradGuard();
        ~NoGradGuard();

    private:
        bool prev_;
    };
    NoGradGuard no_grad();
} // namespace synara