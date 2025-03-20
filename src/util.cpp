#include <iostream>
#include <fstream>
#include <random>

#include "util.hpp"

namespace util {

std::string getParentFolderPath(const std::string &path) {
    if (path.empty()) return "";

    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        std::cerr << "Warning. The input path has no parent folder" << std::endl;
    }
    const std::string directory = (pos == std::string::npos) ? "" : path.substr(0, pos + 1);
    return directory;
}

std::string getFileName(const std::string &path) {
    if (path.empty()) return "";

    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        std::cerr << "Warning. The input path has no parent folder" << std::endl;
    }
    const std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
    return filename;
}

std::string getFileSuffix(const std::string &filename) {
    size_t pos = filename.find_last_of("."); // 查找最后一个 '.'
    if (pos != std::string::npos) {
        return filename.substr(pos); // 截取后缀
    }
    return ""; // 如果没有找到，则返回空字符串
}

std::string iterateOneWordFromLine(const std::string &line, int &wordIter) {
    const int begin = wordIter;
    while (wordIter < line.size() &&
        (line[wordIter] != ' ' && line[wordIter] != '\t' && line[wordIter] != '\r')) {
        ++wordIter;
    }
    const int end = wordIter;

    // Skip the space
    while (wordIter < line.size() &&
        (line[wordIter] == ' ' || line[wordIter] == '\t' || line[wordIter] == '\r')) {
        ++wordIter;
    }

    return end - begin > 0 ? line.substr(begin, end - begin) : "";
}

template<typename T>
bool getDenseMatrixFromFile(const std::string &filePath1, const std::string &filePath2,
                            std::vector<T> &data1, std::vector<T> &data2) {
    std::ifstream inFile1;
    inFile1.open(filePath1, std::ios::in); // open file
    if (!inFile1.is_open()) {
        std::cout << "Error, file1 cannot be opened." << std::endl;
        return false;
    }
    std::ifstream inFile2;
    inFile2.open(filePath2, std::ios::in); // open file
    if (!inFile2.is_open()) {
        std::cout << "Error, file2 cannot be opened." << std::endl;
        return false;
    }

    int wordIter = 0;

    std::string line1; // Store the data for each line
    while (getline(inFile1, line1)) { // line iterator
        wordIter = 0;
        while (wordIter < line1.size()) { // word iterator
            T data = (T) std::stod(iterateOneWordFromLine(line1, wordIter));
            data1.push_back(data);
        }
    }

    std::string line2; // Store the data for each line
    while (getline(inFile2, line2)) { // line iterator
        wordIter = 0;
        while (wordIter < line2.size()) { // word iterator
            T data = (T) std::stod(iterateOneWordFromLine(line2, wordIter));
            data2.push_back(data);
        }
    }

    return true;
}

double truncateFloat(double value, int decimalPlaces = 3) {
    double factor = std::pow(10, decimalPlaces);
    return std::floor(value * factor) / factor;
}

} // namespace util