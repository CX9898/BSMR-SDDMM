#pragma  once

#include <iostream>
#include <string>
#include <vector>

enum MatrixStorageOrder {
  row_major,
  col_major
};

enum MatrixMultiplicationOrder {
  Left_multiplication,
  right_multiplication
};

template<typename T>
class Matrix;

template<typename T>
class SparseMatrix;

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

  SparseMatrix(size_t row, size_t col, size_t nnz) : _row(row), _col(col), _nnz(nnz) {}
  SparseMatrix(size_t row, size_t col, size_t nnz, const std::vector<int> &rowIndex, const std::vector<int> &colIndex)
      : _row(row), _col(col), _nnz(nnz), _rowIndex(rowIndex), _colIndex(colIndex) { _values.resize(nnz); }

  /**
   * Initialize from MatrixMarket file.
   *
   * MatrixMarket file format:
   *    1) The first line describes the file format.
   *    2) The second line has three numbers separated by a space: number of rows, number of columns, and number of non-zeros.
   *    3) Each after line has three numbers separated by a space: current row, current column, and value.
   **/
  bool initializeFromMatrixMarketFile(const std::string &filePath);

  bool initializeValuesFromDenseMatrix(const Matrix<T> &matrixS);

  void printfValue();

  size_t nnz() const {
      return _nnz;
  }

  size_t row() const {
      return _row;
  }
  size_t col() const {
      return _col;
  }

  const std::vector<int> &rowIndex() const {
      return _rowIndex;
  }
  const std::vector<int> &colIndex() const {
      return _colIndex;
  }
  const std::vector<T> &values() const {
      return _values;
  }

  std::vector<T> &setValues() {
      return _values;
  }

 private:
  size_t _row;
  size_t _col;
  size_t _nnz;

  std::vector<int> _rowIndex;
  std::vector<int> _colIndex;
  std::vector<T> _values;
};

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
      : _row(row),
        _col(col),
        _size(size),
        _storageOrder(matrixOrder),
        _leadingDimension(leadingDimension) { _values.resize(size); }

  Matrix(size_t row,
         size_t col,
         size_t size,
         MatrixStorageOrder matrixOrder,
         size_t leadingDimension,
         const std::vector<T> &values)
      : _row(row),
        _col(col),
        _size(size),
        _storageOrder(matrixOrder),
        _leadingDimension(leadingDimension),
        _values(values) {}

  bool initializeFromSparseMatrix(const SparseMatrix<T> &matrixS);
  void changeStorageOrder();

  /**
   * getOneValueForMultiplication
   * Input whether to be used as left or right multiplication in matrix multiplication,
   * the number of rows and columns in which the multiplication is performed and the current iteration k
   **/
  T getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder, size_t row, size_t col, size_t k) const;

  // TODO : Overloads operator <<
  void printfValue();

  size_t size() const {
      return _size;
  }
  MatrixStorageOrder matrixOrder() const {
      return _storageOrder;
  }
  size_t ld() const {
      return _leadingDimension;
  }
  size_t row() const {
      return _row;
  }
  size_t col() const {
      return _col;
  }
  const std::vector<T> &values() const {
      return _values;
  }

//  void setSize(size_t size) {
//      _size = size;
//      _values.resize(size);
//  }
//  void setld(size_t ld) {
//      _leadingDimension = ld;
//  }
//  void setRow(size_t row) {
//      _row = row;
//  }
//  void setCol(size_t col) {
//      _row = col;
//  }
  std::vector<T> &setValues() {
      return _values;
  }

 private:
  size_t _row;
  size_t _col;
  size_t _size;
  MatrixStorageOrder _storageOrder;
  size_t _leadingDimension;

  std::vector<T> _values;
};