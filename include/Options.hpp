#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "util.hpp"

const std::string filePath("../scripts/dataset_of_suiteSparse/bcsstk30/bcsstk30.mtx");

class Options {
public:
    Options(const int argc, const char *const argv[]);

    std::string programPath() const { return programPath_; }
    std::string programName() const { return programName_; }
    std::string inputFile() const { return inputFile_; }
    size_t K() const { return K_; }
    int numIterations() const { return numIterations_; }
    float similarityThresholdAlpha() const { return similarityThresholdAlpha_; }
    float blockDensityThresholdDelta() const { return blockDensityThresholdDelta_; }

private:
    std::string programPath_;
    std::string programName_;
    std::string inputFile_ = filePath;
    size_t K_ = 32;
    int numIterations_ = 10;
    float similarityThresholdAlpha_ = 0.3f;
    float blockDensityThresholdDelta_ = 0.3f;

    inline void parsingOptionAndParameters(const std::string &option,
                                           const std::string &value);
};

inline void Options::parsingOptionAndParameters(const std::string &option,
                                                const std::string &value) {
    try {
        if (option == "-F" || option == "-f") {
            inputFile_ = value;
        }
        if (option == "-K" || option == "-k") {
            K_ = std::stoi(value);
        }
        if (option == "-A" || option == "-a") {
            similarityThresholdAlpha_ = std::stof(value);
        }
        if (option == "-D" || option == "-d") {
            blockDensityThresholdDelta_ = std::stof(value);
        }
    } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
    } catch (const std::out_of_range &e) {
        std::cerr << "Out of range: " << e.what() << std::endl;
    }
}

inline Options::Options(const int argc, const char *const argv[]) {
    programPath_ = util::getParentFolderPath(argv[0]);
    programName_ = util::getFileName(argv[0]);

    // Record the index of the options
    std::vector<int> optionIndices;
    for (int argIdx = 1; argIdx < argc; ++argIdx) {
        if (argv[argIdx][0] == '-') {
            optionIndices.push_back(argIdx);
        }
    }

    // Check options
    std::unordered_map<std::string, std::string> optionToArgumentMap;
    for (const int optionIndex: optionIndices) {
        std::string option_str = argv[optionIndex];

        // Check if the option is duplicated
        if (optionToArgumentMap.find(option_str) != optionToArgumentMap.end()) {
            std::cerr << "Option " << option_str << "is duplicated." << std::endl;
            continue;
        }

        // Check if the option has an argument
        if (optionIndex + 1 >= argc) {
            std::cerr << "Option " << option_str << "requires an argument."
                    << std::endl;
            continue;
        }

        // Record the option and its argument
        const std::string value = argv[optionIndex + 1];
        optionToArgumentMap[option_str] = value;
    }

    // Parsing options
    for (const auto &optionArgumentPair: optionToArgumentMap) {
        parsingOptionAndParameters(optionArgumentPair.first,
                                   optionArgumentPair.second);
    }

    // If no options are provided, use the default input file and K
    if (optionToArgumentMap.empty() && argc > 1) {
        inputFile_ = argv[1];
        K_ = std::stoi(argv[2]);
    }
}
