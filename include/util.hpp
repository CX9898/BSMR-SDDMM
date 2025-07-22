#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <random>

namespace util{
/**
 * @funcitonName: createRandomUniformDistribution
 * @functionInterpretation: Checks if the file exists
 * @input:
 *  `filename` : File name to check
 * @output:
 * Return true if the file exists and false if the file does not
 **/
inline bool fileExists(const std::string& filename){
    std::ifstream file(filename);
    file.close();
    return file.good();
}

template <typename T>
inline std::string to_trimmed_string(T value, int precision = 6);

/**
 * @funcitonName: getFolderPath
 * @functionInterpretation:
 *  Enter the file path or folder path, and return the parent folder path.
 * @input:
 *  `path` : File path.
 * @output:
 *  Returns the parent folder path of the input path.
 **/
inline std::string getParentFolderPath(const std::string& path);

/**
 * @funcitonName: getFileName
 * @functionInterpretation:
 *  Enter the file path or folder path, and return the file name.
 * @input:
 *  `path` : File path.
 * @output:
 *  Returns the file name of the input path.
 **/
inline std::string getFileName(const std::string& path);

/**
 * @funcitonName: getFileSuffix
 * @functionInterpretation:
 *  Enter the file path or folder path, and return the file suffix.
 * @input:
 *  `path` : File path.
 * @output:
 *  Returns the file name of the input path.
 **/
inline std::string getFileSuffix(const std::string& filename);

/**
 * @funcitonName: createRandomUniformDistribution
 * @functionInterpretation: Depending on the type passed to the template,
 * create a uniform distribution that is suitable for float-type or int-type.
 * parameter must be use `static_cast`.
 *
 * @input:
 * `min`: minimum value.
 * `max`: maximum value
 * @output:
 * `std::uniform_real_distribution` or `std::uniform_int_distribution`
 **/
inline std::uniform_real_distribution<float> createRandomUniformDistribution(float min, float max){
    return std::uniform_real_distribution<float>(min, max);
}

inline std::uniform_real_distribution<double> createRandomUniformDistribution(double min, double max){
    return std::uniform_real_distribution<double>(min, max);
}

inline std::uniform_int_distribution<int> createRandomUniformDistribution(int min, int max){
    return std::uniform_int_distribution<int>(min, max);
}

inline std::uniform_int_distribution<uint32_t> createRandomUniformDistribution(uint32_t min, uint32_t max){
    return std::uniform_int_distribution<uint32_t>(min, max);
}

inline std::uniform_int_distribution<uint64_t> createRandomUniformDistribution(uint64_t min, uint64_t max){
    return std::uniform_int_distribution<uint64_t>(min, max);
}

/**
 * @funcitonName: iterateOneWordFromLine
 * @functionInterpretation: Traverse one word from the input line.
 * @input:
 * `line`: line to iterate.
 * `wordIter` : Where to start the traversal. Note that the variables change after the function runs!
 * @output:
 * Return one word starting from the input `wordIter`.
 * `wordIter` will also change to the beginning of the next word
**/
inline std::string iterateOneWordFromLine(const std::string& line, int& wordIter);

/**
 * @funcitonName: getDenseMatrixFromFile
 * @functionInterpretation:
 * @input:
 *
 * @output:
 *
**/
template <typename T>
inline bool getDenseMatrixFromFile(const std::string& filePath1,
                                   const std::string& filePath2,
                                   std::vector<T>& data1,
                                   std::vector<T>& data2);

/**
 * @funcitonName: truncateFloat
 * @functionInterpretation: Floating point numbers are truncated to n decimal places
 * @input:
 * `value`: Floating point number to truncate
 * `decimalPlaces`: Number of decimal places to truncate
 * @output:
 * Returns the truncated floating point number
**/
inline double truncateFloat(double value, int decimalPlaces);
} // namespace util

#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <iomanip>

namespace util{
template <typename T>
std::string to_trimmed_string(T value, const int precision){
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;

    std::string s = oss.str();

    // 去除末尾多余的 0 和小数点
    if (s.find('.') != std::string::npos){
        s.erase(s.find_last_not_of('0') + 1);
        if (s.back() == '.') s.pop_back();
    }

    return s;
}

std::string getParentFolderPath(const std::string& path){
    if (path.empty()) return "";

    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos){
        std::cerr << "Warning. The input path has no parent folder" << std::endl;
    }
    const std::string directory = (pos == std::string::npos) ? "" : path.substr(0, pos + 1);
    return directory;
}

std::string getFileName(const std::string& path){
    if (path.empty()) return "";

    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos){
        std::cerr << "Warning. The input path has no parent folder" << std::endl;
    }
    const std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
    return filename;
}

std::string getFileSuffix(const std::string& filename){
    size_t pos = filename.find_last_of("."); // 查找最后一个 '.'
    if (pos != std::string::npos){
        return filename.substr(pos); // 截取后缀
    }
    return ""; // 如果没有找到，则返回空字符串
}

std::string iterateOneWordFromLine(const std::string& line, int& wordIter){
    const int begin = wordIter;
    while (wordIter < line.size() &&
        (line[wordIter] != ' ' && line[wordIter] != '\t' && line[wordIter] != '\r')){
        ++wordIter;
    }
    const int end = wordIter;

    // Skip the space
    while (wordIter < line.size() &&
        (line[wordIter] == ' ' || line[wordIter] == '\t' || line[wordIter] == '\r')){
        ++wordIter;
    }

    return end - begin > 0 ? line.substr(begin, end - begin) : "";
}

template <typename T>
bool getDenseMatrixFromFile(const std::string& filePath1,
                            const std::string& filePath2,
                            std::vector<T>& data1,
                            std::vector<T>& data2){
    std::ifstream inFile1;
    inFile1.open(filePath1, std::ios::in); // open file
    if (!inFile1.is_open()){
        std::cout << "Error, file1 cannot be opened." << std::endl;
        return false;
    }
    std::ifstream inFile2;
    inFile2.open(filePath2, std::ios::in); // open file
    if (!inFile2.is_open()){
        std::cout << "Error, file2 cannot be opened." << std::endl;
        return false;
    }

    int wordIter = 0;

    std::string line1; // Store the data for each line
    while (getline(inFile1, line1)){
        // line iterator
        wordIter = 0;
        while (wordIter < line1.size()){
            // word iterator
            T data = (T)std::stod(iterateOneWordFromLine(line1, wordIter));
            data1.push_back(data);
        }
    }

    std::string line2; // Store the data for each line
    while (getline(inFile2, line2)){
        // line iterator
        wordIter = 0;
        while (wordIter < line2.size()){
            // word iterator
            T data = (T)std::stod(iterateOneWordFromLine(line2, wordIter));
            data2.push_back(data);
        }
    }

    return true;
}

double truncateFloat(double value, int decimalPlaces = 3){
    double factor = std::pow(10, decimalPlaces);
    return std::floor(value * factor) / factor;
}
} // namespace util
