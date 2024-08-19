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

  Matrix(UIN row,
         UIN col,
         UIN size,
         MatrixStorageOrder matrixOrder,
         UIN leadingDimension)
      : row_(row),
        col_(col),
        size_(size),
        storageOrder_(matrixOrder),
        leadingDimension_(leadingDimension) { values_.resize(size); }

  Matrix(UIN row,
         UIN col,
         UIN size,
         MatrixStorageOrder matrixOrder,
         UIN leadingDimension,
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

  UIN rowOfValueIndex(UIN idx) const;
  UIN colOfValueIndex(UIN idx) const;
  T getOneValue(int row, int col) const;

  /**
   * getOneValueForMultiplication
   * Input whether to be used as left or right multiplication in matrix multiplication,
   * the number of rows and columns in which the multiplication is performed and the current iteration k
   **/
  T getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder, UIN row, UIN col, UIN k) const;

  void makeData(UIN numRow, UIN numCol, MatrixStorageOrder storageOrder = MatrixStorageOrder::row_major);

  void print() const;

  UIN size() const {
      return size_;
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

  const T &operator[](UIN idx) const {
      if (idx > size_) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return values_[idx];
  }
  T &operator[](UIN idx) {
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
  UIN row_tensor() const {
      if (!tensorCoreMode_) {
          return 0;
      }
      return row_tensor_;
  }
  UIN col_tensor() const {
      if (!tensorCoreMode_) {
          return 0;
      }
      return col_tensor_;
  }
  UIN UINensor() const {
      if (!tensorCoreMode_) {
          return 0;
      }
      return UINensor_;
  }
  const std::vector<T> &values_tensor() const {
      if (!tensorCoreMode_) {
          //
      }
      return values_tensor_;
  }

 private:
  UIN row_;
  UIN col_;
  UIN size_;
  MatrixStorageOrder storageOrder_;
  UIN leadingDimension_;

  std::vector<T> values_;

  // TODO: Add tensorCoreMode judgements to all function calls
  bool tensorCoreMode_ = false;
  UIN row_tensor_;
  UIN col_tensor_;
  UIN UINensor_;

  // TODO: delete `values_tensor_` and change `values_`
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

  SparseMatrix(UIN row, UIN col, UIN nnz) : row_(row), col_(col), nnz_(nnz) {
      rowIndex_.resize(nnz);
      colIndex_.resize(nnz);
      values_.resize(nnz);
  }
  SparseMatrix(UIN row, UIN col, UIN nnz, const std::vector<UIN> &rowIndex, const std::vector<UIN> &colIndex)
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

  void makeData(const UIN row, const UIN col, const UIN nnz);

  /**
   * input : idx
   * output : row, col, value
   **/
  void getSpareMatrixOneDataByCOO(const int idx, UIN &row, UIN &col, T &value) const;

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
  void openTensorCoreMode();
  void closeTensorCoreMode();
  UIN row_tensor() const {
      if (!tensorCoreMode_) {
          return 0;
      }
      return row_tensor_;
  }
  UIN col_tensor() const {
      if (!tensorCoreMode_) {
          return 0;
      }
      return col_tensor_;
  }

 private:
  UIN row_;
  UIN col_;
  UIN nnz_;

  std::vector<UIN> rowIndex_;
  std::vector<UIN> colIndex_;
  std::vector<T> values_;

  bool tensorCoreMode_ = false;
  UIN row_tensor_;
  UIN col_tensor_;
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