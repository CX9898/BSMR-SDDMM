#include "matrices.h"

#ifndef UTILITIES_H
#define UTILITIES_H

using namespace std;

template <typename T>
double avg(std::vector<T> const &v)
{
    if (v.empty())
    {
        return 0;
    }

    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

#endif