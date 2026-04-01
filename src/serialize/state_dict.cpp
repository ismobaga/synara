#include "synara/serialize/state_dict.hpp"

#include <fstream>
#include <iomanip>
#include <vector>

#include "synara/core/error.hpp"

namespace synara
{

    void save_state_dict(const StateDict &state, const std::string &path)
    {
        std::ofstream out(path);
        if (!out)
        {
            throw ValueError("save_state_dict(): unable to open file '" + path + "'.");
        }

        out << std::setprecision(17);
        out << "SYNARA_STATE_V1\n";
        out << state.size() << "\n";

        for (const auto &[key, tensor] : state)
        {
            out << key << " " << tensor.rank();
            for (std::size_t d = 0; d < tensor.rank(); ++d)
            {
                out << " " << tensor.shape()[d];
            }
            out << " " << tensor.numel();
            for (std::size_t i = 0; i < tensor.numel(); ++i)
            {
                out << " " << tensor.data()[i];
            }
            out << "\n";
        }
    }

    StateDict load_state_dict(const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
        {
            throw ValueError("load_state_dict(): unable to open file '" + path + "'.");
        }

        std::string magic;
        in >> magic;
        if (magic != "SYNARA_STATE_V1")
        {
            throw ValueError("load_state_dict(): invalid checkpoint format.");
        }

        std::size_t count = 0;
        in >> count;

        StateDict out;
        for (std::size_t item = 0; item < count; ++item)
        {
            std::string key;
            std::size_t rank = 0;
            in >> key >> rank;

            std::vector<std::size_t> dims(rank, 0);
            for (std::size_t d = 0; d < rank; ++d)
            {
                in >> dims[d];
            }

            std::size_t numel = 0;
            in >> numel;
            std::vector<Tensor::value_type> values(numel, 0.0f);
            for (std::size_t i = 0; i < numel; ++i)
            {
                in >> values[i];
            }

            if (!in)
            {
                throw ValueError("load_state_dict(): malformed checkpoint contents.");
            }

            Tensor tensor = Tensor::from_vector(Shape(std::move(dims)), std::move(values), false);
            out.emplace(std::move(key), std::move(tensor));
        }

        return out;
    }

} // namespace synara
