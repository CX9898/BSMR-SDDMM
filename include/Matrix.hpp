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
  Matrix() = default;
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

  bool initializeFromSparseMatrix(const SparseMatrix<T> &matrixS);
  void changeStorageOrder();

  T getOneValue(int row, int col) const;

  /**
   * getOneValueForMultiplication
   * Input whether to be used as left or right multiplication in matrix multiplication,
   * the number of rows and columns in which the multiplication is performed and the current iteration k
   **/
  T getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder, size_t row, size_t col, size_t k) const;

  void makeData(size_t numRow, size_t numCol, MatrixStorageOrder storageOrder = MatrixStorageOrder::row_major);

  // TODO : Overloads operator <<
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

  std::vector<T> &setValues() {
      return values_;
  }

 private:
  size_t row_;
  size_t col_;
  size_t size_;
  MatrixStorageOrder storageOrder_;
  size_t leadingDimension_;

  std::vector<T> values_;
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

  SparseMatrix(size_t row, size_t col, size_t nnz) : row_(row), col_(col), nnz_(nnz) {}
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
  bool initializeFromMatrixMarketFile(const std::string &filePath);

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

  // TODO : Overloads operator <<
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

 private:
  size_t row_;
  size_t col_;
  size_t nnz_;

  std::vector<UIN> rowIndex_;
  std::vector<UIN> colIndex_;
  std::vector<T> values_;
};

namespace dev {

} // namespace dev