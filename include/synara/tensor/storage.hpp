#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace synara
{

    class Storage
    {
    public:
        using value_type = double;

        Storage();
        explicit Storage(std::size_t size);
        explicit Storage(std::vector<value_type> values);

        std::size_t size() const noexcept;

        value_type *data() noexcept;
        const value_type *data() const noexcept;

        value_type &operator[](std::size_t idx);
        const value_type &operator[](std::size_t idx) const;

        std::shared_ptr<std::vector<value_type>> shared_values() const noexcept;

    private:
        std::shared_ptr<std::vector<value_type>> values_;
    };

} // namespace synara
