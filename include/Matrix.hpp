#pragma  once

#include <iostream>
#include <string>
#include <vector>

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

  T getOneValue(int row, int col) const;

  /**
   * getOneValueForMultiplication
   * Input whether to be used as left or right multiplication in matrix multiplication,
   * the number of rows and columns in which the multiplication is performed and the current iteration k
   **/
  T getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder, size_t row, size_t col, size_t k) const;

  void makeData(int row, size_t col, MatrixStorageOrder storageOrder = MatrixStorageOrder::row_major);

  // TODO : Overloads operator <<
  void print();

  size_t size() const {
      return _size;
  }
  MatrixStorageOrder storageOrder() const {
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

  bool setValuesFromMatrix(const Matrix<T> &inputMatrix);

  void makeData(const int row, const int col, const int nnz);

  /**
   * input : idx
   * output : row, col, value
   **/
  void getSpareMatrixOneDataByCOO(const int idx, size_t &row, size_t &col, T &value) const;

  // TODO : Overloads operator <<
  void print();

  /**
   * Used as a test comparison result
   **/
  bool outputToMarketMatrixFile(const std::string &fileName);

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

namespace dev {

} // namespace dev