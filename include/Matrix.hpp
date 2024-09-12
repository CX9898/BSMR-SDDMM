#pragma  once

#include <iostream>
#include <string>
#include <vector>

#include "devVector.cuh"
#include "TensorCoreConfig.cuh"

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

  Matrix(const SparseMatrix<T> &matrixS);

  bool initializeValue(const std::vector<T> &src);
  void changeStorageOrder();

  size_t rowOfValueIndex(size_t idx) const;
  size_t colOfValueIndex(size_t idx) const;
  T getOneValue(size_t row, size_t col) const;

  /**
   * getOneValueForMultiplication
   * Input whether to be used as left or right multiplication in matrix multiplication,
   * the number of rows and columns in which the multiplication is performed and the current iteration k
   **/
  T getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder,
                                 size_t rowMtxC,
                                 size_t colMtxC,
                                 size_t positionOfKIter) const;

  void makeData(size_t numRow, size_t numCol, MatrixStorageOrder storageOrder = MatrixStorageOrder::row_major);

  void print() const;
  void printToMarkdownTable() const;

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
  const std::vector<T> &values() const {
      return values_;
  }
  const T *data() const {
      return values_.data();
  }

  const T &operator[](size_t idx) const {
      if (idx > values_.size()) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return values_[idx];
  }
  T &operator[](size_t idx) {
      if (idx > values_.size()) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return values_[idx];
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

  std::vector<T> values_;

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

  SparseMatrix(size_t row, size_t col, size_t nnz) : row_(row), col_(col), nnz_(nnz) {
      rowIndex_.resize(nnz);
      colIndex_.resize(nnz);
      values_.resize(nnz);
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }
  SparseMatrix(size_t row,
               size_t col,
               size_t nnz,
               const std::vector<size_t> &rowIndex,
               const std::vector<size_t> &colIndex)
      : row_(row), col_(col), nnz_(nnz), rowIndex_(rowIndex), colIndex_(colIndex) {
      values_.resize(nnz);
      if (rowIndex.size() != colIndex.size()) {
          std::cout << "Warning! SparseMatrix initialization error!" << std::endl;
      }
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }
  SparseMatrix(size_t row,
               size_t col,
               size_t nnz,
               const std::vector<size_t> &rowIndex,
               const std::vector<size_t> &colIndex,
               const std::vector<T> &values)
      : row_(row), col_(col), nnz_(nnz), rowIndex_(rowIndex), colIndex_(colIndex), values_(values) {
      if (rowIndex.size() != colIndex.size() != values.size()) {
          std::cout << "Warning! SparseMatrix initialization error!" << std::endl;
      }
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }

  /**
   * Initialize from MatrixMarket file.
   *
   * MatrixMarket file format:
   *    1) The first line describes the file format.
   *    2) The second line has three numbers separated by a space: number of rows, number of columns, and number of non-zeros.
   *    3) Each after line has three numbers separated by a space: current row, current column, and value.
   **/
  void initializeFromMatrixMarketFile(const std::string &filePath);

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
  void getSpareMatrixOneDataByCOO(const size_t idx, size_t &row, size_t &col, T &value) const;

  inline float getSparsity() const {
      return static_cast<float>(row_ * col_ - nnz_) / (row_ * col_);
  }

  void print() const;

  size_t nnz() const {
      return nnz_;
  }

  size_t row() const {
      return row_;
  }
  size_t col() const {
      return col_;
  }

  const std::vector<size_t> &rowIndex() const {
      return rowIndex_;
  }
  const std::vector<size_t> &colIndex() const {
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

  /**
   * tensor core mode
   **/
  void openTensorCoreMode(MatrixMultiplicationOrder multiplicationOrder);
  void openTensorCoreModeForSampled(TensorCoreConfig tensorCoreConfig);
  void closeTensorCoreMode();
  const std::vector<size_t> &matrixTileIndex() {
      return matrixTileIndexForTensorCore_;
  }

 private:
  size_t row_;
  size_t col_;
  size_t nnz_;

  std::vector<size_t> rowIndex_;
  std::vector<size_t> colIndex_;
  std::vector<T> values_;

  bool tensorCoreMode_ = false;
  std::vector<size_t> matrixTileIndexForTensorCore_;
  size_t rowBeforeChange_;
  size_t colBeforeChange_;
  std::vector<size_t> rowIndexBeforeChange_;
  std::vector<size_t> colIndexBeforeChange_;
  std::vector<T> valuesBeforeChange_;
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

  Matrix(const SparseMatrix<T> &matrixS);

  bool initializeValue(const std::vector<T> &src);
  void changeStorageOrder();

  size_t rowOfValueIndex(size_t idx) const;
  size_t colOfValueIndex(size_t idx) const;
  T getOneValue(size_t row, size_t col) const;

  /**
   * getOneValueForMultiplication
   * Input whether to be used as left or right multiplication in matrix multiplication,
   * the number of rows and columns in which the multiplication is performed and the current iteration k
   **/
  T getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder,
                                 size_t rowMtxC,
                                 size_t colMtxC,
                                 size_t positionOfKIter) const;

  void makeData(size_t numRow, size_t numCol, MatrixStorageOrder storageOrder = MatrixStorageOrder::row_major);

  void print() const;
  void printToMarkdownTable() const;

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

//  const T &operator[](size_t idx) const {
//      if (idx > values_.size()) {
//          std::cerr << "Error! Array access out of bounds" << std::endl;
//      }
//      return values_[idx];
//  }
//  T &operator[](size_t idx) {
//      if (idx > values_.size()) {
//          std::cerr << "Error! Array access out of bounds" << std::endl;
//      }
//      return values_[idx];
//  }

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

  SparseMatrix(size_t row, size_t col, size_t nnz) : row_(row), col_(col), nnz_(nnz) {
      rowIndex_.resize(nnz);
      colIndex_.resize(nnz);
      values_.resize(nnz);
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }
  SparseMatrix(size_t row,
               size_t col,
               size_t nnz,
               const dev::vector<size_t> &rowIndex,
               const dev::vector<size_t> &colIndex)
      : row_(row), col_(col), nnz_(nnz), rowIndex_(rowIndex), colIndex_(colIndex) {
      values_.resize(nnz);
      if (rowIndex.size() != colIndex.size()) {
          std::cout << "Warning! SparseMatrix initialization error!" << std::endl;
      }
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }
  SparseMatrix(size_t row,
               size_t col,
               size_t nnz,
               const std::vector<size_t> &rowIndex,
               const std::vector<size_t> &colIndex,
               const std::vector<T> &values)
      : row_(row), col_(col), nnz_(nnz), rowIndex_(rowIndex), colIndex_(colIndex), values_(values) {
      if (rowIndex.size() != colIndex.size() != values.size()) {
          std::cout << "Warning! SparseMatrix initialization error!" << std::endl;
      }
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }

  /**
   * Initialize from MatrixMarket file.
   *
   * MatrixMarket file format:
   *    1) The first line describes the file format.
   *    2) The second line has three numbers separated by a space: number of rows, number of columns, and number of non-zeros.
   *    3) Each after line has three numbers separated by a space: current row, current column, and value.
   **/
  void initializeFromMatrixMarketFile(const std::string &filePath);

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
  void getSpareMatrixOneDataByCOO(const size_t idx, size_t &row, size_t &col, T &value) const;

  inline float getSparsity() const {
      return static_cast<float>(row_ * col_ - nnz_) / (row_ * col_);
  }

  void print() const;

  size_t nnz() const {
      return nnz_;
  }

  size_t row() const {
      return row_;
  }
  size_t col() const {
      return col_;
  }

  const dev::vector<size_t> &rowIndex() const {
      return rowIndex_;
  }
  const dev::vector<size_t> &colIndex() const {
      return colIndex_;
  }
  const dev::vector<T> &values() const {
      return values_;
  }

  dev::vector<T> &setValues() {
      return values_;
  }

//  const T &operator[](size_t idx) const {
//      if (idx > nnz_) {
//          std::cerr << "Error! Array access out of bounds" << std::endl;
//      }
//      return values_[idx];
//  }
//  T &operator[](size_t idx) {
//      if (idx > nnz_) {
//          std::cerr << "Error! Array access out of bounds" << std::endl;
//      }
//      return values_[idx];
//  }

  /**
   * tensor core mode
   **/
  void openTensorCoreMode(MatrixMultiplicationOrder multiplicationOrder);
  void openTensorCoreModeForSampled(TensorCoreConfig tensorCoreConfig);
  void closeTensorCoreMode();
  const dev::vector<size_t> &matrixTileIndex() {
      return matrixTileIndexForTensorCore_;
  }

 private:
  size_t row_;
  size_t col_;
  size_t nnz_;

  dev::vector<size_t> rowIndex_;
  dev::vector<size_t> colIndex_;
  dev::vector<T> values_;

  bool tensorCoreMode_ = false;
  dev::vector<size_t> matrixTileIndexForTensorCore_;
  size_t rowBeforeChange_;
  size_t colBeforeChange_;
  dev::vector<size_t> rowIndexBeforeChange_;
  dev::vector<size_t> colIndexBeforeChange_;
  dev::vector<T> valuesBeforeChange_;
};
} // namespace dev