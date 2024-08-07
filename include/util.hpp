#pragma once

#include <string>
#include <vector>

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