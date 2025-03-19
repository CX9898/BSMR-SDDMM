#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

//const std::string folderPath("../../dataset/");
//const std::string fileName = ("nips");
const std::string folderPath("../../../dataset/test/matrix_20000_20000_/");
const std::string fileName = ("matrix_20000_20000_4000000");
const std::string fileFormat(".mtx");
//const std::string filePath = folderPath + fileName + fileFormat;
const std::string filePath("./../../../autoTestTool/dataset_of_isratnisa_paper_results/dataset/web-NotreDame.txt");

class Options {
 public:
  Options(int argc, char *argv[]);

  std::string inputFile() const { return inputFile_; }
  size_t k() const { return k_; }
  float alpha() const { return alpha_; }
  float beta() const { return beta_; }
  int tile_sizeX() const { return tile_sizeX_; }
  int tile_sizeY() const { return tile_sizeY_; }

 private:
  std::string program_name_;
  std::string inputFile_ = filePath;
  size_t k_ = 32;
  float alpha_ = 1.0f;
  float beta_ = 0.0f;
  int tile_sizeX_ = 50000;
  int tile_sizeY_ = 192;

  std::unordered_set<std::string> shortOptions_ = {
      "-A", "-a",
      "-B", "-b",
      "-F", "-f",
      "-K", "-k",
      "-X",
      "-Y"
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
            k_ = std::stoi(value);
        }
        if (option == "-X") {
            tile_sizeX_ = std::stoi(value);
        }
        if (option == "-Y") {
            tile_sizeY_ = std::stoi(value);
        }
    } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
    } catch (const std::out_of_range &e) {
        std::cerr << "Out of range: " << e.what() << std::endl;
    }
}

std::string getParentFolderPath(const std::string &path) {
    for (int idx = path.size() - 2; idx >= 0; --idx) {
        if (path[idx] == '/' || path[idx] == '\\') {
            return path.substr(0, idx + 1);
        }
    }
    std::cerr << "Warning. The input path has no parent folder" << std::endl;
    return path;
}

Options::Options(int argc, char **argv) {
    program_name_ = getParentFolderPath(argv[0]) + argv[0];

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