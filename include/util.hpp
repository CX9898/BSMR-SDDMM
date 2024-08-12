#pragma once

#include <string>
#include <vector>


/**
 * namespace io
 * Including file IO functions
 **/
namespace io {

/**
 * @funcitonName: createRandomUniformDistribution
 * @functionInterpretation: Checks if the file exists
 * @input:
 *
 * @output:
 *
 **/
inline bool fileExists(const std::string &filename) {
    std::ifstream file(filename);
    file.close();
    return file.good();
}

} // namespace io

/**
 * @funcitonName: createRandomUniformDistribution
 * @functionInterpretation: Depending on the type passed to the template,
 * create a uniform distribution that is suitable for float-type or int-type
 * @input:
 * `min`: minimum value.
 * `max`: maximum value
 * @output:
 * `std::uniform_real_distribution` or `std::uniform_int_distribution`
 **/
template<typename T>
std::uniform_real_distribution<T> createRandomUniformDistribution(T min, T max);
template<typename T>
std::uniform_int_distribution<int> createRandomUniformDistribution(int min, int max);
template<typename T>
std::uniform_int_distribution<uint64_t> createRandomUniformDistribution(uint64_t min, uint64_t max);

template<typename T>
T getRandomData(std::mt19937 &generator, T min, T max);

/**
 * @funcitonName: iterateOneWordFromLine
 * @functionInterpretation:
 * @input:
 * `line`: line to iterate.
 * `wordIter` : Where to start the traversal, Note that the variables change after the function runs!
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