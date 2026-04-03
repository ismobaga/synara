#include "synara/serialize/state_dict.hpp"

#include <fstream>
#include <iomanip>
#include <vector>
#include <cstring>

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

    // Binary checkpoint format
    void save_state_dict_binary(const StateDict &state, const std::string &path)
    {
        std::ofstream out(path, std::ios::binary);
        if (!out)
        {
            throw ValueError("save_state_dict_binary(): unable to open file '" + path + "'.");
        }

        // Write header
        const char *header = "SYNARA_BINARY_V1";
        out.write(header, 16);

        // Write format version
        uint32_t version = 1;
        out.write(reinterpret_cast<const char *>(&version), sizeof(uint32_t));

        // Write number of tensors
        uint32_t num_tensors = static_cast<uint32_t>(state.size());
        out.write(reinterpret_cast<const char *>(&num_tensors), sizeof(uint32_t));

        for (const auto &[key, tensor] : state)
        {
            // Write key string
            uint32_t key_len = static_cast<uint32_t>(key.size());
            out.write(reinterpret_cast<const char *>(&key_len), sizeof(uint32_t));
            out.write(key.data(), key_len);

            // Write rank
            uint32_t rank = static_cast<uint32_t>(tensor.rank());
            out.write(reinterpret_cast<const char *>(&rank), sizeof(uint32_t));

            // Write dimensions
            for (std::size_t d = 0; d < tensor.rank(); ++d)
            {
                uint64_t dim = static_cast<uint64_t>(tensor.shape()[d]);
                out.write(reinterpret_cast<const char *>(&dim), sizeof(uint64_t));
            }

            // Write number of elements
            uint64_t numel = static_cast<uint64_t>(tensor.numel());
            out.write(reinterpret_cast<const char *>(&numel), sizeof(uint64_t));

            // Write tensor data
            const double *data = tensor.data();
            out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(numel * sizeof(double)));
        }

        if (!out)
        {
            throw ValueError("save_state_dict_binary(): error writing to file.");
        }
    }

    StateDict load_state_dict_binary(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            throw ValueError("load_state_dict_binary(): unable to open file '" + path + "'.");
        }

        // Read and verify header
        char header[16];
        in.read(header, 16);
        if (!in || std::string(header, 16) != "SYNARA_BINARY_V1")
        {
            throw ValueError("load_state_dict_binary(): invalid binary checkpoint format.");
        }

        // Read version
        uint32_t version = 0;
        in.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
        if (!in || version != 1)
        {
            throw ValueError("load_state_dict_binary(): unsupported checkpoint version.");
        }

        // Read number of tensors
        uint32_t num_tensors = 0;
        in.read(reinterpret_cast<char *>(&num_tensors), sizeof(uint32_t));
        if (!in)
        {
            throw ValueError("load_state_dict_binary(): error reading checkpoint header.");
        }

        StateDict out;
        for (uint32_t item = 0; item < num_tensors; ++item)
        {
            // Read key string
            uint32_t key_len = 0;
            in.read(reinterpret_cast<char *>(&key_len), sizeof(uint32_t));
            if (!in)
            {
                throw ValueError("load_state_dict_binary(): error reading key length.");
            }

            std::string key(key_len, '\0');
            in.read(&key[0], key_len);
            if (!in)
            {
                throw ValueError("load_state_dict_binary(): error reading key.");
            }

            // Read rank
            uint32_t rank = 0;
            in.read(reinterpret_cast<char *>(&rank), sizeof(uint32_t));
            if (!in)
            {
                throw ValueError("load_state_dict_binary(): error reading rank.");
            }

            // Read dimensions
            std::vector<std::size_t> dims(rank, 0);
            for (uint32_t d = 0; d < rank; ++d)
            {
                uint64_t dim = 0;
                in.read(reinterpret_cast<char *>(&dim), sizeof(uint64_t));
                if (!in)
                {
                    throw ValueError("load_state_dict_binary(): error reading dimension.");
                }
                dims[d] = static_cast<std::size_t>(dim);
            }

            // Read number of elements
            uint64_t numel = 0;
            in.read(reinterpret_cast<char *>(&numel), sizeof(uint64_t));
            if (!in)
            {
                throw ValueError("load_state_dict_binary(): error reading numel.");
            }

            // Read tensor data
            std::vector<double> values(static_cast<std::size_t>(numel), 0.0);
            in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(numel * sizeof(double)));
            if (!in)
            {
                throw ValueError("load_state_dict_binary(): error reading tensor data.");
            }

            Tensor tensor = Tensor::from_vector(Shape(std::move(dims)), std::move(values), false);
            out.emplace(std::move(key), std::move(tensor));
        }

        return out;
    }

} // namespace synara
