#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "synara/visualization/plot.hpp"

int main()
{
    using namespace synara;

    const std::vector<PlotSeries> series = {
        PlotSeries{"train_loss", {1.0, 0.8, 0.6, 0.4}, '*'},
        PlotSeries{"eval_loss", {1.1, 0.9, 0.7, 0.5}, 'o'},
    };

    PlotOptions options;
    options.title = "Loss curves";
    options.width = 32;
    options.height = 10;

    const std::string ascii = render_line_plot(series, options);
    assert(ascii.find("Loss curves") != std::string::npos);
    assert(ascii.find("train_loss") != std::string::npos);
    assert(ascii.find("eval_loss") != std::string::npos);
    assert(ascii.find('*') != std::string::npos || ascii.find('o') != std::string::npos);

    const std::filesystem::path svg_path = "synara_plot_test.svg";
    assert(write_line_plot_svg(series, svg_path.string(), options));

    std::ifstream in(svg_path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string svg = buffer.str();
    assert(svg.find("<svg") != std::string::npos);
    assert(svg.find("train_loss") != std::string::npos);
    assert(svg.find("eval_loss") != std::string::npos);

    std::filesystem::remove(svg_path);
    return 0;
}
