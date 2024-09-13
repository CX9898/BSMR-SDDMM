#pragma once

#include "Matrix.hpp"

namespace dev {
/**
 * The default is row-major order, but if you want to switch to column-major order, call the changeMajorOrder function.
 **/
template<typename T>
class Matrix {
 public:
  Matrix() = delete;
  ~Matrix() = default;

  Matrix(size_t row,
         size_t col,
         MatrixStorageOrder matrixOrder)
      : row_(row),
        col_(col),
        storageOrder_(matrixOrder) {
      leadingDimension_ = matrixOrder == MatrixStorageOrder::row_major ? col : row;
      values_.resize(row * col);
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }

  Matrix(size_t row,
         size_t col,
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

  size_t rowOfValueIndex(size_t idx) const;
  size_t colOfValueIndex(size_t idx) const;

  size_t size() const {
      return values_.size();
  }
  MatrixStorageOrder storageOrder() const {
      return storageOrder_;
  }
  size_t ld() const {
      return leadingDimension_;
  }
  size_t row() const {
      return row_;
  }
  size_t col() const {
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
  size_t row_;
  size_t col_;
  MatrixStorageOrder storageOrder_ = row_major;
  size_t leadingDimension_;

  dev::vector<T> values_;

  bool tensorCoreMode_ = false;
  size_t rowBeforeChange_;
  size_t colBeforeChange_;
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
  void openTensorCoreMode(MatrixMultiplicationOrder multiplicationOrder);
  void openTensorCoreModeForSampled(TensorCoreConfig tensorCoreConfig);
  void closeTensorCoreMode();
  const dev::vector<UIN> &matrixTileIndex() const {
      return matrixTileIndex_;
  }
  const dev::vector<UIN> &tileIndexPerWar() const {
      return tileIndexPerWarp_;
  }

 private:
  UIN row_;
  UIN col_;
  UIN nnz_;

  dev::vector<UIN> rowIndex_;
  dev::vector<UIN> colIndex_;
  dev::vector<T> values_;

  bool tensorCoreMode_ = false;
  dev::vector<UIN> matrixTileIndex_;
  dev::vector<UIN> tileIndexPerWarp_;

  UIN rowBeforeChange_;
  UIN colBeforeChange_;
};
} // namespace dev