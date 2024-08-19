#pragma  once

#include <iostream>
#include <string>
#include <vector>

using UIN = uint64_t;

enum MatrixStorageOrder {
  row_major,
  col_major
};

enum MatrixMultiplicationOrder {
  left_multiplication,
  right_multiplication
};

template<typename T>
class Matrix;

template<typename T>
class SparseMatrix;

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
         size_t size,
         MatrixStorageOrder matrixOrder,
         size_t leadingDimension)
      : row_(row),
        col_(col),
        size_(size),
        storageOrder_(matrixOrder),
        leadingDimension_(leadingDimension) { values_.resize(size); }

  Matrix(size_t row,
         size_t col,
         size_t size,
         MatrixStorageOrder matrixOrder,
         size_t leadingDimension,
         const std::vector<T> &values)
      : row_(row),
        col_(col),
        size_(size),
        storageOrder_(matrixOrder),
        leadingDimension_(leadingDimension),
        values_(values) {}

  Matrix(const SparseMatrix<T> &matrixS);

  bool initializeValue(const std::vector<T> &src);
  void changeStorageOrder();

  size_t rowOfValueIndex(size_t idx);
  size_t colOfValueIndex(size_t idx);
  T getOneValue(int row, int col) const;

  /**
   * getOneValueForMultiplication
   * Input whether to be used as left or right multiplication in matrix multiplication,
   * the number of rows and columns in which the multiplication is performed and the current iteration k
   **/
  T getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder, size_t row, size_t col, size_t k) const;

  void makeData(size_t numRow, size_t numCol, MatrixStorageOrder storageOrder = MatrixStorageOrder::row_major);

  void print();

  size_t size() const {
      return size_;
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
  const std::vector<T> &values() const {
      return values_;
  }

  const T &operator[](size_t idx) const {
      if (idx > size_) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return values_[idx];
  }
  T &operator[](size_t idx) {
      if (idx > size_) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return values_[idx];
  }

  /**
   * tensor core mode
   **/
  void openTensorCoreMode();
  void closeTensorCoreMode();
  size_t row_tensor() const {
      if (!tensorCoreMode_) {
          return 0;
      }
      return row_tensor_;
  }
  size_t col_tensor() const {
      if (!tensorCoreMode_) {
          return 0;
      }
      return col_tensor_;
  }
  size_t size_tensor() const {
      if (!tensorCoreMode_) {
          return 0;
      }
      return size_tensor_;
  }
  const std::vector<T> &values_tensor() const {
      if (!tensorCoreMode_) {
          return std::vector<T>(0);
      }
      return values_tensor_;
  }

 private:
  size_t row_;
  size_t col_;
  size_t size_;
  MatrixStorageOrder storageOrder_;
  size_t leadingDimension_;

  std::vector<T> values_;

  bool tensorCoreMode_ = false;
  size_t row_tensor_;
  size_t col_tensor_;
  size_t size_tensor_;

  std::vector<T> values_tensor_;
};

/**
 * SparseMatrix class
 *
 * Store in COO format.
 **/
template<typename T>
class SparseMatrix {
 public:
  SparseMatrix() = delete;
  ~SparseMatrix() = default;

  SparseMatrix(size_t row, size_t col, size_t nnz) : row_(row), col_(col), nnz_(nnz) {
      rowIndex_.resize(nnz);
      colIndex_.resize(nnz);
      values_.resize(nnz);
  }
  SparseMatrix(size_t row, size_t col, size_t nnz, const std::vector<UIN> &rowIndex, const std::vector<UIN> &colIndex)
      : row_(row), col_(col), nnz_(nnz), rowIndex_(rowIndex), colIndex_(colIndex) { values_.resize(nnz); }

  /**
   * Initialize from MatrixMarket file.
   *
   * MatrixMarket file format:
   *    1) The first line describes the file format.
   *    2) The second line has three numbers separated by a space: number of rows, number of columns, and number of non-zeros.
   *    3) Each after line has three numbers separated by a space: current row, current column, and value.
   **/
  SparseMatrix(const std::string &filePath);

  /**
    * Used as a test comparison result
    **/
  bool outputToMarketMatrixFile(const std::string &fileName);

  bool setValuesFromMatrix(const Matrix<T> &inputMatrix);

  void makeData(const size_t row, const size_t col, const size_t nnz);

  /**
   * input : idx
   * output : row, col, value
   **/
  void getSpareMatrixOneDataByCOO(const int idx, size_t &row, size_t &col, T &value) const;

  void print();

  size_t nnz() const {
      return nnz_;
  }

  size_t row() const {
      return row_;
  }
  size_t col() const {
      return col_;
  }

  const std::vector<UIN> &rowIndex() const {
      return rowIndex_;
  }
  const std::vector<UIN> &colIndex() const {
      return colIndex_;
  }
  const std::vector<T> &values() const {
      return values_;
  }

  std::vector<T> &setValues() {
      return values_;
  }

  const T &operator[](size_t idx) const {
      if (idx > nnz_) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return values_[idx];
  }
  T &operator[](size_t idx) {
      if (idx > nnz_) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return values_[idx];
  }

 private:
  size_t row_;
  size_t col_;
  size_t nnz_;

  std::vector<UIN> rowIndex_;
  std::vector<UIN> colIndex_;
  std::vector<T> values_;

  size_t row_tensor_;
  size_t col_tensor_;
  size_t nnz_tensor_;
};

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const Matrix<T> &mtx) {
    os << " [row : " << mtx.row() << ", col : " << mtx.col() << "]";
    return os;
}

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const SparseMatrix<T> &mtxS) {
    os << " [row : " << mtxS.row() << ", col : " << mtxS.col() << ", nnz : " << mtxS.nnz() << "]";
    return os;
}

namespace dev {

} // namespace dev