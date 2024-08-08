#pragma once

#include <string>
#include <vector>


/**
 * Including file IO functions
 **/
namespace io {

/**
 * Checks if the file exists
 **/
inline bool fileExists(const std::string &filename) {
    std::ifstream file(filename);
    file.close();
    return file.good();
}
} // namespace io


/**
 *
 **/
std::string iterateOneWordFromLine(const std::string &line, int &wordIter);

/**
 *
 **/
template<typename T>
bool getDenseMatrixFromFile(const std::string &filePath1, const std::string &filePath2,
                            std::vector<T> &data1, std::vector<T> &data2);