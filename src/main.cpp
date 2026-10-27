#include "Simulation.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        std::cout << "Initialize" << std::endl;
#ifdef cNoC_MODE
        std::cout << "NOTE: Running in cNoC_MODE" << std::endl;
#endif
        SimulationOptions options;
        std::string json_path, csv_path;
        if (argc == 1) std::cout << "Running with default parameters" << std::endl;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::cout << "NoCDASim [-NNmodel FILE] [-NNweight FILE] [-NNinput FILE] "
                             "[--max-cycles N] [--report-json FILE] [--report-csv FILE]\n";
                return 0;
            }
            if (i + 1 == argc) throw std::invalid_argument("Missing value for " + flag);
            const std::string value = argv[++i];
            if (flag == "-NNmodel") options.model_file = value;
            else if (flag == "-NNweight") options.weight_file = value;
            else if (flag == "-NNinput") options.input_file = value;
            else if (flag == "--report-json") json_path = value;
            else if (flag == "--report-csv") csv_path = value;
            else if (flag == "--max-cycles") {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("Invalid --max-cycles: " + value);
                options.max_cycles = std::stoull(value);
            } else throw std::invalid_argument("Invalid option: " + flag);
        }
        const auto begin = std::chrono::steady_clock::now();
        Simulation simulation(options);
        const auto status = simulation.run();
        simulation.reportLegacy();
        if (!json_path.empty()) {
            std::ofstream out(json_path);
            if (!out) throw std::runtime_error("Cannot open JSON report: " + json_path);
            simulation.writeJson(out);
            out.close();
            if (!out) throw std::runtime_error("Cannot write JSON report: " + json_path);
        }
        if (!csv_path.empty()) {
            std::ofstream out(csv_path);
            if (!out) throw std::runtime_error("Cannot open CSV report: " + csv_path);
            simulation.writeLayerCsv(out);
            out.close();
            if (!out) throw std::runtime_error("Cannot write CSV report: " + csv_path);
        }
        const auto end = std::chrono::steady_clock::now();
        std::cout << "Execution time (sec) = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0
                  << std::endl;
        std::cout << "!!END!!" << std::endl;
        return status == StopReason::Completed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}
