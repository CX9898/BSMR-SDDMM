#pragma once

#include "Matrix.hpp"
#include "devVector.cuh"

namespace dev {
/**
 * The default is row-major order, but if you want to switch to column-major order, call the changeMajorOrder function.
 **/
template<typename T>
class Matrix {
 public:
  Matrix() = delete;
  ~Matrix() = default;

  Matrix(UIN row,
         UIN col,
         MatrixStorageOrder matrixOrder)
      : row_(row),
        col_(col),
        storageOrder_(matrixOrder) {
      leadingDimension_ = matrixOrder == MatrixStorageOrder::row_major ? col : row;
      values_.resize(row * col);
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }

  Matrix(UIN row,
         UIN col,
         MatrixStorageOrder matrixOrder,
         const std::vector<T> &values)
      : row_(row),
        col_(col),
        storageOrder_(matrixOrder),
        values_(values) {
      leadingDimension_ = matrixOrder == MatrixStorageOrder::row_major ? col : row;
      if (row * col != values.size()) {
          std::cout << "Warning! Matrix initialization mismatch" << std::endl;
      }
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }

  UIN rowOfValueIndex(UIN idx) const;
  UIN colOfValueIndex(UIN idx) const;

  UIN size() const {
      return values_.size();
  }
  MatrixStorageOrder storageOrder() const {
      return storageOrder_;
  }
  UIN ld() const {
      return leadingDimension_;
  }
  UIN row() const {
      return row_;
  }
  UIN col() const {
      return col_;
  }
  const dev::vector<T> &values() const {
      return values_;
  }
  dev::vector<T> &setValues() {
      return values_;
  }
  const T *data() const {
      return values_.data();
  }

  /**
   * tensor core mode
   **/
  void openTensorCoreMode(MatrixMultiplicationOrder multiplicationOrder);
  void closeTensorCoreMode();

 private:
  UIN row_;
  UIN col_;
  MatrixStorageOrder storageOrder_ = row_major;
  UIN leadingDimension_;

  dev::vector<T> values_;

  bool tensorCoreMode_ = false;
  UIN rowBeforeChange_;
  UIN colBeforeChange_;
};

/**
 * SparseMatrix class
 *
 * Store in COO format.
 **/
template<typename T>
class SparseMatrix {
 public:
  SparseMatrix() = default;
  ~SparseMatrix() = default;

  SparseMatrix(UIN row, UIN col, UIN nnz) : row_(row), col_(col), nnz_(nnz) {
      rowIndex_.resize(nnz);
      colIndex_.resize(nnz);
      values_.resize(nnz);
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }
  SparseMatrix(::SparseMatrix<T> src)
      : row_(src.row()),
        col_(src.col()),
        nnz_(src.nnz()),
        rowIndex_(src.rowIndex()),
        colIndex_(src.colIndex()),
        values_(src.values()) {};

  /**
   * Initialize from MatrixMarket file.
   *
   * MatrixMarket file format:
   *    1) The first line describes the file format.
   *    2) The second line has three numbers separated by a space: number of rows, number of columns, and number of non-zeros.
   *    3) Each after line has three numbers separated by a space: current row, current column, and value.
   **/
  void initializeFromMatrixMarketFile(const std::string &filePath);

  inline float getSparsity() const {
      return static_cast<float>(row_ * col_ - nnz_) / (row_ * col_);
  }

  size_t nnz() const {
      return nnz_;
  }

  size_t row() const {
      return row_;
  }
  size_t col() const {
      return col_;
  }

  const dev::vector<UIN> &rowIndex() const {
      return rowIndex_;
  }
  const dev::vector<UIN> &colIndex() const {
      return colIndex_;
  }
  const dev::vector<T> &values() const {
      return values_;
  }

  dev::vector<T> &setValues() {
      return values_;
  }

  /**
   * tensor core mode
   **/
  void openTensorCoreModeForSampled(TensorCoreConfig tensorCoreConfig);
  void closeTensorCoreMode();
  const dev::vector<UIN> &matrixTileMappedToWarpIndex() const {
      return matrixTileMappedToWarpIndex_;
  }
  const dev::vector<UIN> &matrixTileMappedToWarpIndexData() const {
      return matrixTileMappedToWarpIndexData_;
  }

 private:
  UIN row_;
  UIN col_;
  UIN nnz_;

  dev::vector<UIN> rowIndex_;
  dev::vector<UIN> colIndex_;
  dev::vector<T> values_;

  bool tensorCoreMode_ = false;
  dev::vector<UIN> matrixTileMappedToWarpIndex_;
  dev::vector<UIN> matrixTileMappedToWarpIndexData_;

  UIN rowBeforeChange_;
  UIN colBeforeChange_;
};
} // namespace dev