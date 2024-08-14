#include <iostream>
#include <fstream>
#include <random>

#include "util.hpp"

namespace util {

template<>
float getRandomData(std::mt19937 &generator, float min, float max) {
    std::uniform_real_distribution<float> distribution(min, max);

    return distribution(generator);
}

template<>
int getRandomData(std::mt19937 &generator, int min, int max) {
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}

template<>
uint64_t getRandomData(std::mt19937 &generator, uint64_t min, uint64_t max) {
    std::uniform_int_distribution<uint64_t> distribution(min, max);
    return distribution(generator);
}

std::string iterateOneWordFromLine(const std::string &line, int &wordIter) {
    const int begin = wordIter;
    while (wordIter < line.size() && line[wordIter] != ' ') {
        ++wordIter;
    }
    const int end = wordIter;
    ++wordIter;

    return line.substr(begin, end - begin);
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

} // namespace util