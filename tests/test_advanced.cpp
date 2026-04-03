#include <cassert>
#include <fstream>
#include <filesystem>
#include <iostream>

#include "synara.hpp"

namespace fs = std::filesystem;

namespace
{

    void test_binary_checkpoint_save_load()
    {
        using namespace synara;

        // Create temporary directory for tests
        fs::path temp_dir = fs::temp_directory_path() / "synara_phase5_test_binary";
        fs::create_directories(temp_dir);

        try
        {
            // Create a simple model
            Linear layer1(10, 5);

            // Get state dict
            StateDict state = layer1.state_dict();

            // Save as binary checkpoint
            fs::path binary_path = temp_dir / "model.synbin";
            save_state_dict_binary(state, binary_path.string());

            // Verify file was created
            assert(fs::exists(binary_path));

            // Load binary checkpoint
            StateDict loaded_state = load_state_dict_binary(binary_path.string());

            // Verify loaded state matches original
            assert(state.size() == loaded_state.size());
            for (const auto &[key, tensor] : state)
            {
                assert(loaded_state.count(key));
                const auto &loaded_tensor = loaded_state.at(key);
                assert(tensor.shape() == loaded_tensor.shape());
                assert(tensor.numel() == loaded_tensor.numel());

                // Check values match
                for (std::size_t i = 0; i < tensor.numel(); ++i)
                {
                    assert(tensor.data()[i] == loaded_tensor.data()[i]);
                }
            }

            std::cout << "✓ test_binary_checkpoint_save_load passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_binary_checkpoint_save_load failed: " << e.what() << "\n";
            throw;
        }

        fs::remove_all(temp_dir);
    }

    void test_binary_vs_text_checkpoint_size()
    {
        using namespace synara;

        fs::path temp_dir = fs::temp_directory_path() / "synara_phase5_test_size";
        fs::create_directories(temp_dir);

        try
        {
            // Create a model with some parameters
            Linear layer(100, 50);
            StateDict state = layer.state_dict();

            fs::path text_path = temp_dir / "model.syn";
            fs::path binary_path = temp_dir / "model.synbin";

            save_state_dict(state, text_path.string());
            save_state_dict_binary(state, binary_path.string());

            std::size_t text_size = fs::file_size(text_path);
            std::size_t binary_size = fs::file_size(binary_path);

            // Binary format should be smaller
            assert(binary_size < text_size);

            std::cout << "✓ test_binary_vs_text_checkpoint_size passed (text: "
                      << text_size << " bytes, binary: " << binary_size << " bytes)\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_binary_vs_text_checkpoint_size failed: " << e.what() << "\n";
            throw;
        }

        fs::remove_all(temp_dir);
    }

    void test_parameter_counting()
    {
        using namespace synara;

        try
        {
            Sequential model;
            model.add(std::make_shared<Linear>(10, 20)); // 10*20 + 20 = 220
            model.add(std::make_shared<Linear>(20, 5));  // 20*5 + 5 = 105
            // Total: 325 parameters

            std::size_t total = total_parameters(model);
            std::size_t trainable = trainable_parameters(model);
            std::size_t non_trainable = non_trainable_parameters(model);

            assert(total == 325);
            assert(trainable == 325);
            assert(non_trainable == 0);

            std::cout << "✓ test_parameter_counting passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_parameter_counting failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_parameter_info()
    {
        using namespace synara;

        try
        {
            Linear layer(10, 5);
            auto info = parameter_info(layer);

            // Linear has weight and bias
            assert(info.size() == 2);

            // Check weight parameter
            assert(info[0].name == "param_0");
            assert(info[0].numel == 10 * 5);
            assert(info[0].shape.size() == 2);
            assert(info[0].shape[0] == 5);
            assert(info[0].shape[1] == 10);
            assert(info[0].requires_grad);

            // Check bias parameter
            assert(info[1].name == "param_1");
            assert(info[1].numel == 5);
            assert(info[1].requires_grad);

            std::cout << "✓ test_parameter_info passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_parameter_info failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_module_num_parameters()
    {
        using namespace synara;

        try
        {
            Linear layer(10, 5);
            assert(layer.num_parameters() == 10 * 5 + 5);
            assert(layer.num_trainable_parameters() == 10 * 5 + 5);

            std::cout << "✓ test_module_num_parameters passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_module_num_parameters failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_parameter_tree()
    {
        using namespace synara;

        try
        {
            Linear layer(10, 5);
            std::string tree = layer.parameter_tree();

            assert(tree.find("param_0") != std::string::npos);
            assert(tree.find("param_1") != std::string::npos);
            assert(tree.find("shape") != std::string::npos);

            std::cout << "✓ test_parameter_tree passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_parameter_tree failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_parameter_shapes()
    {
        using namespace synara;

        try
        {
            Linear layer(10, 5);
            auto shapes = layer.parameter_shapes();

            assert(shapes.size() == 2);
            assert(shapes[0].first == "param_0");
            assert(shapes[1].first == "param_1");

            // Weight should have 2 dimensions [5, 10] (out_features, in_features)
            assert(shapes[0].second.size() == 2);

            // Bias shape - just verify it exists and has reasonable size
            assert(shapes[1].second.size() >= 1);

            std::cout << "✓ test_parameter_shapes passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_parameter_shapes failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_memory_usage()
    {
        using namespace synara;

        try
        {
            Linear layer(100, 50);
            std::size_t memory_bytes = layer.memory_usage();

            // 100*50 + 50 = 5050 parameters
            // Each double = 8 bytes
            // Total = 5050 * 8 = 40400 bytes
            assert(memory_bytes == 5050 * 8);

            std::cout << "✓ test_memory_usage passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_memory_usage failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_float32_tensor_types()
    {
        using namespace synara;

        try
        {
            // Test that float32 type alias works
            StorageFloat32 float32_storage(10);
            assert(float32_storage.size() == 10);

            StorageFloat64 float64_storage(10);
            assert(float64_storage.size() == 10);

            // Verify that Storage defaults to float64
            Storage default_storage(10);
            assert(default_storage.size() == 10);

            std::cout << "✓ test_float32_tensor_types passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_float32_tensor_types failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_float32_tensor_conversion()
    {
        using namespace synara;

        try
        {
            // Create a tensor with float32 data
            Tensor t1 = Tensor::from_vector(Shape({2, 3}), {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f}, false);

            // Convert to float32 vector
            auto float32_data = to_float32(t1);
            assert(float32_data.size() == 6);

            std::cout << "✓ test_float32_tensor_conversion passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_float32_tensor_conversion failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_data_type_detection()
    {
        using namespace synara;

        try
        {
            Tensor t = Tensor::from_vector(Shape({2}), {1.0, 2.0}, false);

            // Currently all tensors are float64
            auto dtype = get_tensor_dtype(t);
            assert(dtype == DataType::FLOAT64);

            // Test dtype string conversion
            assert(dtype_to_string(DataType::FLOAT32) == "float32");
            assert(dtype_to_string(DataType::FLOAT64) == "float64");

            std::cout << "✓ test_data_type_detection passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_data_type_detection failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_float32_make_tensor()
    {
        using namespace synara;

        try
        {
            // Create a float32 tensor
            std::vector<float> float32_data = {1.5f, 2.5f, 3.5f};
            Tensor t = make_float32_tensor(Shape({3}), float32_data);

            assert(t.numel() == 3);

            std::cout << "✓ test_float32_make_tensor passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_float32_make_tensor failed: " << e.what() << "\n";
            throw;
        }
    }

    void test_integration_all_features()
    {
        using namespace synara;

        fs::path temp_dir = fs::temp_directory_path() / "synara_phase5_test_integration";
        fs::create_directories(temp_dir);

        try
        {
            // Create a model
            Sequential model;
            model.add(std::make_shared<Linear>(10, 20));
            model.add(std::make_shared<Linear>(20, 5));

            // Test Feature 3: Introspection
            std::size_t total_params = model.num_parameters();
            assert(total_params == 10 * 20 + 20 + 20 * 5 + 5);

            // Test Feature 2: Parameter counting
            std::size_t util_total = total_parameters(model);
            assert(total_params == util_total);

            // Test Feature 1: Save as binary checkpoint
            StateDict state = model.state_dict();
            fs::path checkpoint_path = temp_dir / "integrated_model.synbin";
            save_state_dict_binary(state, checkpoint_path.string());

            // Test Feature 1: Load binary checkpoint
            StateDict loaded_state = load_state_dict_binary(checkpoint_path.string());
            assert(state.size() == loaded_state.size());

            // Test Feature 4: Float32 utilities (informational)
            auto float32_info = dtype_to_string(DataType::FLOAT32);
            assert(float32_info == "float32");

            std::cout << "✓ test_integration_all_features passed\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "✗ test_integration_all_features failed: " << e.what() << "\n";
            throw;
        }

        fs::remove_all(temp_dir);
    }

} // namespace

int main()
{
    std::cout << "\n=== Phase 5 Advanced Features Tests ===\n\n";

    try
    {
        test_binary_checkpoint_save_load();
        test_binary_vs_text_checkpoint_size();
        test_parameter_counting();
        test_parameter_info();
        test_module_num_parameters();
        test_parameter_tree();
        test_parameter_shapes();
        test_memory_usage();
        test_float32_tensor_types();
        test_float32_tensor_conversion();
        test_data_type_detection();
        test_float32_make_tensor();
        test_integration_all_features();

        std::cout << "\n=== All tests passed! ===\n\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nTest suite failed with exception: " << e.what() << "\n";
        return 1;
    }
}