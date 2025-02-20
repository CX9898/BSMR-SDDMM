#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

const std::string folderPath("../dataset/test/matrix_20000_20000_/");
//const std::string folderPath("./");
//const std::string fileName = ("nips");
//const std::string fileName = ("test2");
const std::string fileName("matrix_20000_20000_20000000");
const std::string fileFormat(".mtx");
const std::string filePath = folderPath + fileName + fileFormat;

class Options {
 public:
  Options(int argc, char *argv[]);

  std::string inputFile() const { return inputFile_; }
  size_t K() const { return K_; }
  float alpha() const { return alpha_; }
  float beta() const { return beta_; }

 private:
  std::string program_name_;
  std::string inputFile_ = filePath;
  size_t K_ = 32;
  float alpha_ = 1.0f;
  float beta_ = 0.0f;

  std::unordered_set<std::string> shortOptions_ = {
      "-A", "-a",
      "-B", "-b",
      "-F", "-f",
      "-K", "-k"
  };

  inline void parsingOptionAndParameters(const std::string &option, const std::string &value);
};

inline void Options::parsingOptionAndParameters(const std::string &option, const std::string &value) {
    try {
        if (option == "-A" || option == "-a") {
            alpha_ = std::stof(value);
        }
        if (option == "-B" || option == "-b") {
            beta_ = std::stof(value);
        }
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

Options::Options(int argc, char **argv) {
    program_name_ = util::getParentFolderPath(argv[0]) + argv[0];

    // 记录参数的索引
    std::vector<int> optionIndices;
    for (int argIdx = 1; argIdx < argc; ++argIdx) {
        if (argv[argIdx][0] == '-') {
            optionIndices.push_back(argIdx);
        }
    }

    // 检查是否有重复的参数

    // 解析参数
    for (int index : optionIndices) {
        std::string option_str = argv[index];

        if (shortOptions_.find(option_str) == shortOptions_.end()) {
            std::cerr << "Unknown option: " << option_str.substr(1, option_str.size() - 1) << std::endl;
            continue;
        }

        if (index + 1 >= argc) {
            std::cerr << "Option " << option_str << " requires an argument." << std::endl;
            continue;
        }

        parsingOptionAndParameters(option_str, argv[index + 1]);
    }
}