#pragma  once

#include <iostream>
#include <string>
#include <vector>

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

  Matrix(const SparseMatrix<T> &matrixS);

  bool initializeValue(const std::vector<T> &src);
  void changeStorageOrder();

  UIN rowOfValueIndex(UIN idx) const;
  UIN colOfValueIndex(UIN idx) const;
  T getOneValue(UIN row, UIN col) const;

  /**
   * getOneValueForMultiplication
   * Input whether to be used as left or right multiplication in matrix multiplication,
   * the number of rows and columns in which the multiplication is performed and the current iteration k
   **/
  T getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder,
                                 UIN rowMtxC,
                                 UIN colMtxC,
                                 UIN positionOfKIter) const;

  void makeData(UIN numRow, UIN numCol, MatrixStorageOrder storageOrder = MatrixStorageOrder::row_major);

  void print() const;
  void printToMarkdownTable() const;

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
  const std::vector<T> &values() const {
      return values_;
  }
  const T *data() const {
      return values_.data();
  }

  const T &operator[](UIN idx) const {
      if (idx > values_.size()) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return values_[idx];
  }
  T &operator[](UIN idx) {
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
  UIN row_;
  UIN col_;
  MatrixStorageOrder storageOrder_ = row_major;
  UIN leadingDimension_;

  std::vector<T> values_;

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
  SparseMatrix(UIN row,
               UIN col,
               UIN nnz,
               const std::vector<UIN> &rowIndex,
               const std::vector<UIN> &colIndex)
      : row_(row), col_(col), nnz_(nnz), rowIndex_(rowIndex), colIndex_(colIndex) {
      values_.resize(nnz);
      if (rowIndex.size() != colIndex.size()) {
          std::cout << "Warning! SparseMatrix initialization error!" << std::endl;
      }
      rowBeforeChange_ = row;
      colBeforeChange_ = col;
  }
  SparseMatrix(UIN row,
               UIN col,
               UIN nnz,
               const std::vector<UIN> &rowIndex,
               const std::vector<UIN> &colIndex,
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

  void makeData(const UIN row, const UIN col, const UIN nnz);

  /**
   * input : idx
   * output : row, col, value
   **/
  void getSpareMatrixOneDataByCOO(const UIN idx, UIN &row, UIN &col, T &value) const;

  inline float getSparsity() const {
      return static_cast<float>(row_ * col_ - nnz_) / (row_ * col_);
  }

  void print() const;

  UIN nnz() const {
      return nnz_;
  }

  UIN row() const {
      return row_;
  }
  UIN col() const {
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

  const T &operator[](UIN idx) const {
      if (idx > nnz_) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return values_[idx];
  }
  T &operator[](UIN idx) {
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
  const std::vector<UIN> &matrixTileIndex() {
      return matrixTileIndexForTensorCore_;
  }

 private:
  UIN row_;
  UIN col_;
  UIN nnz_;

  std::vector<UIN> rowIndex_;
  std::vector<UIN> colIndex_;
  std::vector<T> values_;

  bool tensorCoreMode_ = false;
  std::vector<UIN> matrixTileIndexForTensorCore_;
  UIN rowBeforeChange_;
  UIN colBeforeChange_;
  std::vector<UIN> rowIndexBeforeChange_;
  std::vector<UIN> colIndexBeforeChange_;
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