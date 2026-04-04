#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "synara/core/types.hpp"

namespace synara
{

    struct PlotSeries
    {
        std::string label;
        std::vector<double> values;
        char glyph = '*';
    };

    struct PlotOptions
    {
        Size width = 60;
        Size height = 12;
        std::string title;
        bool show_legend = true;
        double y_min = std::numeric_limits<double>::quiet_NaN();
        double y_max = std::numeric_limits<double>::quiet_NaN();
    };

    namespace plot_detail
    {
        inline std::pair<double, double> value_range(const std::vector<PlotSeries> &series, const PlotOptions &options)
        {
            double min_value = std::numeric_limits<double>::infinity();
            double max_value = -std::numeric_limits<double>::infinity();

            for (const auto &s : series)
            {
                for (const double value : s.values)
                {
                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                }
            }

            if (std::isfinite(options.y_min))
            {
                min_value = options.y_min;
            }
            if (std::isfinite(options.y_max))
            {
                max_value = options.y_max;
            }

            if (!std::isfinite(min_value) || !std::isfinite(max_value))
            {
                min_value = 0.0;
                max_value = 1.0;
            }
            if (min_value == max_value)
            {
                min_value -= 1.0;
                max_value += 1.0;
            }

            return {min_value, max_value};
        }

        inline void put_cell(std::vector<std::string> &grid, int x, int y, char glyph)
        {
            if (y < 0 || x < 0 || y >= static_cast<int>(grid.size()) || x >= static_cast<int>(grid[0].size()))
            {
                return;
            }

            char &cell = grid[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
            if (cell == ' ' || cell == glyph)
            {
                cell = glyph;
            }
            else
            {
                cell = '#';
            }
        }

        inline void draw_line(std::vector<std::string> &grid, int x0, int y0, int x1, int y1, char glyph)
        {
            int dx = std::abs(x1 - x0);
            const int sx = x0 < x1 ? 1 : -1;
            int dy = -std::abs(y1 - y0);
            const int sy = y0 < y1 ? 1 : -1;
            int err = dx + dy;

            for (;;)
            {
                put_cell(grid, x0, y0, glyph);
                if (x0 == x1 && y0 == y1)
                {
                    break;
                }

                const int e2 = 2 * err;
                if (e2 >= dy)
                {
                    err += dy;
                    x0 += sx;
                }
                if (e2 <= dx)
                {
                    err += dx;
                    y0 += sy;
                }
            }
        }

        inline std::string svg_escape(const std::string &value)
        {
            std::string escaped;
            escaped.reserve(value.size());
            for (const char c : value)
            {
                switch (c)
                {
                case '&':
                    escaped += "&amp;";
                    break;
                case '<':
                    escaped += "&lt;";
                    break;
                case '>':
                    escaped += "&gt;";
                    break;
                case '"':
                    escaped += "&quot;";
                    break;
                default:
                    escaped.push_back(c);
                    break;
                }
            }
            return escaped;
        }

        inline const char *color_for_index(std::size_t index)
        {
            static constexpr std::array<const char *, 8> palette = {
                "#2563eb", "#dc2626", "#16a34a", "#9333ea",
                "#ea580c", "#0891b2", "#ca8a04", "#475569"};
            return palette[index % palette.size()];
        }

        inline bool has_data(const std::vector<PlotSeries> &series)
        {
            for (const auto &s : series)
            {
                if (!s.values.empty())
                {
                    return true;
                }
            }
            return false;
        }

    } // namespace plot_detail

    inline std::string render_line_plot(const std::vector<PlotSeries> &series, const PlotOptions &options = {})
    {
        std::ostringstream oss;
        if (!options.title.empty())
        {
            oss << options.title << "\n";
        }

        if (!plot_detail::has_data(series))
        {
            oss << "(no data)";
            return oss.str();
        }

        const std::size_t plot_width = std::max<std::size_t>(2, options.width);
        const std::size_t plot_height = std::max<std::size_t>(2, options.height);
        std::vector<std::string> grid(plot_height, std::string(plot_width, ' '));

        const auto [min_value, max_value] = plot_detail::value_range(series, options);

        for (const auto &s : series)
        {
            if (s.values.empty())
            {
                continue;
            }

            int prev_x = -1;
            int prev_y = -1;
            for (std::size_t i = 0; i < s.values.size(); ++i)
            {
                const int x = (s.values.size() == 1)
                                  ? 0
                                  : static_cast<int>(std::lround(
                                        static_cast<double>(i) * static_cast<double>(plot_width - 1) /
                                        static_cast<double>(s.values.size() - 1)));

                const double normalized = std::clamp(
                    (s.values[i] - min_value) / (max_value - min_value),
                    0.0,
                    1.0);
                const int y = static_cast<int>(plot_height - 1) -
                              static_cast<int>(std::lround(normalized * static_cast<double>(plot_height - 1)));

                if (prev_x >= 0 && prev_y >= 0)
                {
                    plot_detail::draw_line(grid, prev_x, prev_y, x, y, s.glyph);
                }
                else
                {
                    plot_detail::put_cell(grid, x, y, s.glyph);
                }

                prev_x = x;
                prev_y = y;
            }
        }

        oss << std::fixed << std::setprecision(3);
        for (std::size_t row = 0; row < plot_height; ++row)
        {
            const double row_value = max_value -
                                     static_cast<double>(row) * (max_value - min_value) /
                                         static_cast<double>(plot_height - 1);
            const bool show_label = row == 0 || row == plot_height / 2 || row + 1 == plot_height;
            if (show_label)
            {
                oss << std::setw(8) << row_value;
            }
            else
            {
                oss << "        ";
            }
            oss << " |" << grid[row] << "|\n";
        }

        oss << "         +" << std::string(plot_width, '-') << "+\n";
        oss << "          1";
        if (plot_width > 2)
        {
            oss << std::string(plot_width - 2, ' ');
        }
        const std::size_t max_points = [&]()
        {
            std::size_t count = 0;
            for (const auto &s : series)
            {
                count = std::max(count, s.values.size());
            }
            return count;
        }();
        oss << max_points << "\n";

        if (options.show_legend)
        {
            oss << "legend: ";
            for (std::size_t i = 0; i < series.size(); ++i)
            {
                if (i > 0)
                {
                    oss << "  ";
                }
                oss << series[i].glyph << '=' << series[i].label;
            }
        }

        return oss.str();
    }

    inline bool write_line_plot_svg(
        const std::vector<PlotSeries> &series,
        const std::string &path,
        const PlotOptions &options = {})
    {
        std::ofstream out(path);
        if (!out)
        {
            return false;
        }

        const int svg_width = std::max<int>(320, static_cast<int>(options.width) * 12 + 120);
        const int svg_height = std::max<int>(220, static_cast<int>(options.height) * 18 + 120);
        const int margin_left = 60;
        const int margin_right = 20;
        const int margin_top = 40;
        const int margin_bottom = 50;
        const int plot_width = svg_width - margin_left - margin_right;
        const int plot_height = svg_height - margin_top - margin_bottom;

        const auto [min_value, max_value] = plot_detail::value_range(series, options);
        std::size_t max_points = 0;
        for (const auto &s : series)
        {
            max_points = std::max(max_points, s.values.size());
        }
        max_points = std::max<std::size_t>(max_points, 1);

        out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << svg_width
            << "\" height=\"" << svg_height << "\" viewBox=\"0 0 " << svg_width << " " << svg_height << "\">\n";
        out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
        if (!options.title.empty())
        {
            out << "<text x=\"" << (svg_width / 2) << "\" y=\"24\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"16\">"
                << plot_detail::svg_escape(options.title) << "</text>\n";
        }

        out << "<line x1=\"" << margin_left << "\" y1=\"" << (margin_top + plot_height)
            << "\" x2=\"" << (margin_left + plot_width) << "\" y2=\"" << (margin_top + plot_height)
            << "\" stroke=\"#0f172a\" stroke-width=\"1.5\"/>\n";
        out << "<line x1=\"" << margin_left << "\" y1=\"" << margin_top
            << "\" x2=\"" << margin_left << "\" y2=\"" << (margin_top + plot_height)
            << "\" stroke=\"#0f172a\" stroke-width=\"1.5\"/>\n";

        out << std::fixed << std::setprecision(3);
        for (int i = 0; i < 5; ++i)
        {
            const double t = static_cast<double>(i) / 4.0;
            const double value = max_value - t * (max_value - min_value);
            const int y = margin_top + static_cast<int>(std::lround(t * plot_height));
            out << "<line x1=\"" << margin_left << "\" y1=\"" << y
                << "\" x2=\"" << (margin_left + plot_width) << "\" y2=\"" << y
                << "\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>\n";
            out << "<text x=\"" << (margin_left - 8) << "\" y=\"" << (y + 4)
                << "\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"11\" fill=\"#334155\">"
                << value << "</text>\n";
        }

        out << "<text x=\"" << margin_left << "\" y=\"" << (svg_height - 12)
            << "\" font-family=\"monospace\" font-size=\"11\" fill=\"#334155\">epoch 1</text>\n";
        out << "<text x=\"" << (margin_left + plot_width) << "\" y=\"" << (svg_height - 12)
            << "\" text-anchor=\"end\" font-family=\"monospace\" font-size=\"11\" fill=\"#334155\">epoch "
            << max_points << "</text>\n";

        for (std::size_t index = 0; index < series.size(); ++index)
        {
            const auto &s = series[index];
            if (s.values.empty())
            {
                continue;
            }

            out << "<polyline fill=\"none\" stroke=\"" << plot_detail::color_for_index(index)
                << "\" stroke-width=\"2\" points=\"";
            for (std::size_t i = 0; i < s.values.size(); ++i)
            {
                const double t_x = (s.values.size() == 1)
                                       ? 0.0
                                       : static_cast<double>(i) / static_cast<double>(s.values.size() - 1);
                const int x = margin_left + static_cast<int>(std::lround(t_x * plot_width));
                const double normalized = std::clamp(
                    (s.values[i] - min_value) / (max_value - min_value),
                    0.0,
                    1.0);
                const int y = margin_top + plot_height -
                              static_cast<int>(std::lround(normalized * plot_height));
                out << x << ',' << y;
                if (i + 1 != s.values.size())
                {
                    out << ' ';
                }
            }
            out << "\"/>\n";

            const int legend_y = margin_top + 16 + static_cast<int>(index) * 18;
            out << "<rect x=\"" << (margin_left + 8) << "\" y=\"" << (legend_y - 10)
                << "\" width=\"10\" height=\"10\" fill=\"" << plot_detail::color_for_index(index) << "\"/>\n";
            out << "<text x=\"" << (margin_left + 24) << "\" y=\"" << legend_y
                << "\" font-family=\"monospace\" font-size=\"12\" fill=\"#0f172a\">"
                << plot_detail::svg_escape(s.label) << "</text>\n";
        }

        out << "</svg>\n";
        return static_cast<bool>(out);
    }

} // namespace synara
