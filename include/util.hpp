#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <random>

namespace util {

/**
 * @funcitonName: createRandomUniformDistribution
 * @functionInterpretation: Checks if the file exists
 * @input:
 *  `filename` : File name to check
 * @output:
 * Return true if the file exists and false if the file does not
 **/
inline bool fileExists(const std::string &filename) {
    std::ifstream file(filename);
    file.close();
    return file.good();
}

/**
 * @funcitonName: getFolderPath
 * @functionInterpretation:
 *  Enter the file path or folder path, and return the parent folder path.
 * @input:
 *  `path` : File path.
 * @output:
 *  Returns the parent folder path of the input path.
 **/
std::string getParentFolderPath(const std::string &path);

/**
 * @funcitonName: getFileName
 * @functionInterpretation:
 *  Enter the file path or folder path, and return the file name.
 * @input:
 *  `path` : File path.
 * @output:
 *  Returns the file name of the input path.
 **/
std::string getFileName(const std::string &path);

/**
 * @funcitonName: getFileSuffix
 * @functionInterpretation:
 *  Enter the file path or folder path, and return the file suffix.
 * @input:
 *  `path` : File path.
 * @output:
 *  Returns the file name of the input path.
 **/
std::string getFileSuffix(const std::string& filename);

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
inline std::uniform_real_distribution<float> createRandomUniformDistribution(float min, float max) {
    return std::uniform_real_distribution<float>(min, max);
}
inline std::uniform_real_distribution<double> createRandomUniformDistribution(double min, double max) {
    return std::uniform_real_distribution<double>(min, max);
}
inline std::uniform_int_distribution<int> createRandomUniformDistribution(int min, int max) {
    return std::uniform_int_distribution<int>(min, max);
}
inline std::uniform_int_distribution<uint32_t> createRandomUniformDistribution(uint32_t min, uint32_t max) {
    return std::uniform_int_distribution<uint32_t>(min, max);
}
inline std::uniform_int_distribution<uint64_t> createRandomUniformDistribution(uint64_t min, uint64_t max) {
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
std::string iterateOneWordFromLine(const std::string &line, int &wordIter);

/**
 * @funcitonName: getDenseMatrixFromFile
 * @functionInterpretation:
 * @input:
 *
 * @output:
 * 
**/
template<typename T>
bool getDenseMatrixFromFile(const std::string &filePath1, const std::string &filePath2,
                            std::vector<T> &data1, std::vector<T> &data2);

/**
 * @funcitonName: truncateFloat
 * @functionInterpretation: Floating point numbers are truncated to n decimal places
 * @input:
 * `value`: Floating point number to truncate
 * `decimalPlaces`: Number of decimal places to truncate
 * @output:
 * Returns the truncated floating point number
**/
double truncateFloat(double value, int decimalPlaces);
}// namespace util