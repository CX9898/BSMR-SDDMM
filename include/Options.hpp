#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "util.hpp"

const std::string folderPath("../dataset/test/matrix_15000_15000_/");
//const std::string folderPath("../autoTestTool/matrices/");
//const std::string folderPath("./");
//const std::string fileName = ("nips");
//const std::string fileName = ("test2");
//const std::string fileName("1138_bus/1138_bus");
const std::string fileName("matrix_15000_15000_11250000");
const std::string fileFormat(".mtx");
const std::string filePath = folderPath + fileName + fileFormat;
//const std::string filePath("./../autoTestTool/dataset_of_isratnisa_paper_results/dataset/loc-gowalla_edges.txt");

class Options {
 public:
  Options(const int argc, const char *const argv[]);

  std::string programPath() const { return programPath_; }
  std::string programName() const { return programName_; }
  std::string inputFile() const { return inputFile_; }
  size_t K() const { return K_; }

 private:
  std::string programPath_;
  std::string programName_;
  std::string inputFile_ = filePath;
  size_t K_ = 512;

  std::unordered_set<std::string> shortOptions_ = {
      "-F", "-f",
      "-K", "-k",
  };
  inline void parsingOptionAndParameters(const std::string &option, const std::string &value);
};

inline void Options::parsingOptionAndParameters(const std::string &option, const std::string &value) {
    try {
        if (option == "-F" || option == "-f") {
            inputFile_ = value;
        }
        if (option == "-K" || option == "-k") {
            K_ = std::stoi(value);
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
    for (int optionIndex : optionIndices) {
        std::string option_str = argv[optionIndex];

        // Check if the option is duplicated
        if (optionToArgumentMap.find(option_str) != optionToArgumentMap.end()) {
            std::cerr << "Option " << option_str << "is duplicated." << std::endl;
            continue;
        }

        // Check if the option is valid
        if (shortOptions_.find(option_str) == shortOptions_.end()) {
            std::cerr << "Unknown option: " << option_str.substr(1, option_str.size() - 1) << std::endl
                      << "Please check the usage of the program." << std::endl;
            continue;
        }

        // Check if the option has an argument
        if (optionIndex + 1 >= argc) {
            std::cerr << "Option " << option_str << "requires an argument." << std::endl;
            continue;
        }

        // Record the option and its argument
        std::string value = argv[optionIndex + 1];
        optionToArgumentMap[option_str] = value;
    }

    // Parsing options
    for (const auto &optionArgumentPair : optionToArgumentMap) {
        parsingOptionAndParameters(optionArgumentPair.first, optionArgumentPair.second);
    }
}