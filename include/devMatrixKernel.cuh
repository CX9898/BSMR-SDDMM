#pragma once

#include "devMatrix.cuh"

__global__ void getNumIndexPerWarp(const UIN size, const UIN numWarpX,
                                   const UIN numTileM, const UIN numTileN,
                                   const UIN nnz,
                                   const UIN *rowIndex,
                                   const UIN *colIndex,
                                   UIN *numIndexPerWarp);

__global__ void getTileIndexDataPerWarp(const UIN size, const UIN numWarpX,
                                        const UIN numTileM, const UIN numTileN,
                                        const UIN nnz,
                                        const UIN *rowIndex,
                                        const UIN *colIndex,
                                        const UIN *matrixTileMappedToWarpIndex,
                                        UIN *matrixTileMappedToWarpIndexData);