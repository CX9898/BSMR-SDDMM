#include <cuda_fp16.h>

#ifndef DEFINITIONS_H
#define DEFINITIONS_H

using intT = int;
using DataT = float;
using DataT_H = __half;
using DataT_C = float;

enum FileFormatType
{
    smtx,
    mtx
};

enum Reordering
{
    none,
    cpu,
    gpu
};

enum Major
{
    row,
    col
};

#endif