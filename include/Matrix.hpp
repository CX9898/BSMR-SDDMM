#pragma  once

#include <iostream>
#include <string>
#include <vector>

enum MatrixOrder {
  row_major,
  col_major
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

  void printfValue() {
      std::cout << "row :\t";
      for (auto iter : _rowIndex) {
          std::cout << iter << "\t";
      }
      std::cout << std::endl;
      std::cout << "col :\t";
      for (auto iter : _colIndex) {
          std::cout << iter << "\t";
      }
      std::cout << std::endl;
      std::cout << "value :\t";
      for (auto iter : _values) {
          std::cout << iter << "\t";
      }
      std::cout << std::endl;
  }

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
         MatrixOrder matrixOrder,
         size_t leadingDimension,
         const std::vector<T> &values)
      : _row(row),
        _col(col),
        _size(size),
        _matrixOrder(matrixOrder),
        _leadingDimension(leadingDimension),
        _values(values) {}

  bool initializeFromSparseMatrix(const SparseMatrix<T> &matrixS);
  void changeMajorOrder();

  void printfValue() {
      for (auto iter : _values) {
          std::cout << iter << " ";
      }
      std::cout << std::endl;
  }

  size_t size() const {
      return _size;
  }
  MatrixOrder matrixOrder() const {
      return _matrixOrder;
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

  void setSize(size_t size) {
      _size = size;
      _values.resize(size);
  }
  void setld(size_t ld) {
      _leadingDimension = ld;
  }
  void setRow(size_t row) {
      _row = row;
  }
  void setCol(size_t col) {
      _row = col;
  }
  std::vector<T> &setValues() {
      return _values;
  }

 private:
  size_t _row;
  size_t _col;
  size_t _size;
  MatrixOrder _matrixOrder;
  size_t _leadingDimension;

  std::vector<T> _values;
};