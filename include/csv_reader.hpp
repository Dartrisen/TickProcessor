#pragma once

#include "dataset.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

class CsvReader
{
public:
    static MarketDataset load(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        if (!std::getline(file, line))
        {
            throw std::runtime_error("Empty CSV file");
        }

        const auto                                   header = parse_line(line);
        std::unordered_map<std::string, std::size_t> columns;
        for (std::size_t i = 0; i < header.size(); ++i)
        {
            columns[std::string(trim(header[i]))] = i;
        }

        const auto index = [&](const std::string& name)
        {
            auto it = columns.find(name);
            if (it == columns.end())
                throw std::runtime_error("Missing CSV column: " + name);
            return it->second;
        };

        const auto datetime_idx = index("datetime");
        const auto open_idx     = index("Open");
        const auto high_idx     = index("High");
        const auto low_idx      = index("Low");
        const auto close_idx    = index("Close");
        const auto volume_idx   = index("Volume");
        const auto source_idx   = index("source");

        MarketDataset dataset;
        std::size_t   line_number = 1;

        while (std::getline(file, line))
        {
            ++line_number;
            if (line.empty())
                continue;

            try
            {
                const auto fields = parse_line(line);
                auto       get    = [&](std::size_t i) -> std::string_view
                {
                    if (i >= fields.size())
                        throw std::runtime_error("Field index out of bounds");
                    return fields[i];
                };

                Tick bar;
                bar.datetime = std::string(get(datetime_idx));
                bar.open     = parse_double(get(open_idx));
                bar.high     = parse_double(get(high_idx));
                bar.low      = parse_double(get(low_idx));
                bar.close    = parse_double(get(close_idx));
                bar.volume   = parse_volume(get(volume_idx));
                bar.source   = std::string(get(source_idx));

                validate(bar);
                dataset.add(std::move(bar));
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error("CSV error at line " + std::to_string(line_number) + ": " + e.what());
            }
        }
        return dataset;
    }

private:
    static std::string_view trim(std::string_view value)
    {
        constexpr std::string_view ws    = " \t\r\n";
        const auto                 first = value.find_first_not_of(ws);
        if (first == std::string_view::npos)
            return {};
        const auto last = value.find_last_not_of(ws);
        return value.substr(first, last - first + 1);
    }

    static std::vector<std::string> parse_line(const std::string& line)
    {
        std::vector<std::string> fields;
        std::string              field;
        bool                     quoted = false;

        for (std::size_t i = 0; i < line.size(); ++i)
        {
            char c = line[i];
            if (c == '"')
            {
                if (quoted && i + 1 < line.size() && line[i + 1] == '"')
                {
                    field += '"';
                    ++i;
                }
                else
                {
                    quoted = !quoted;
                }
            }
            else if (c == ',' && !quoted)
            {
                fields.push_back(std::move(field));
                field.clear();
            }
            else
            {
                field += c;
            }
        }
        if (quoted)
            throw std::runtime_error("Unterminated quote");
        fields.push_back(std::move(field));
        return fields;
    }

    static double parse_double(std::string_view value)
    {
        std::string str(trim(value));
        if (str.empty())
            throw std::runtime_error("Empty numeric field");
        std::size_t pos = 0;
        double      res = std::stod(str, &pos);
        if (pos != str.size() || !std::isfinite(res))
            throw std::runtime_error("Invalid double");
        return res;
    }

    static std::uint64_t parse_volume(std::string_view value)
    {
        std::string str(trim(value));
        if (str.empty() || str.front() == '-')
            throw std::runtime_error("Invalid volume");
        std::size_t pos = 0;
        auto        res = std::stoull(str, &pos);
        if (pos != str.size())
            throw std::runtime_error("Invalid volume format");
        return res;
    }

    static void validate(const Tick& bar)
    {
        if (bar.datetime.empty())
            throw std::runtime_error("Empty timestamp");
        if (bar.open <= 0 || bar.high <= 0 || bar.low <= 0 || bar.close <= 0)
        {
            throw std::runtime_error("Non-positive price detected");
        }
        if (bar.high < std::max(bar.open, bar.close) || bar.low > std::min(bar.open, bar.close))
        {
            throw std::runtime_error("Invalid OHLC dynamic ranges");
        }
    }
};
