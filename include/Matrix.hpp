#pragma  once

#include <iostream>
#include <string>
#include <vector>
#include <tuple>

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

namespace sparseMatrix {
struct DataBase;

template<typename T>
struct CSR;

template<typename T>
struct COO;

template<typename T>
struct BELL;
}

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
  }

  Matrix(UIN row,
         UIN col,
         MatrixStorageOrder matrixOrder,
         const T *values)
      : row_(row),
        col_(col),
        storageOrder_(matrixOrder),
        values_(values, values + row * col) {
      leadingDimension_ = matrixOrder == MatrixStorageOrder::row_major ? col : row;
  }

  Matrix(const sparseMatrix::COO<T> &matrixS);

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

  void makeData();

  void makeData(UIN numRow, UIN numCol);

  void print() const;

  void printToMarkdownTable() const;

  std::vector<T> getRowVector(UIN row) const;

  std::vector<T> getColVector(UIN col) const;

  UIN size() const {
      return values_.size();
  }

  MatrixStorageOrder storageOrder() const {
      return storageOrder_;
  }

  UIN leadingDimension() const {
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

 private:
  UIN row_;
  UIN col_;
  MatrixStorageOrder storageOrder_ = row_major;
  UIN leadingDimension_;

  std::vector<T> values_;
};

template<typename T>
inline std::ostream &operator<<(std::ostream &os, const Matrix<T> &mtx) {
    os << " [row : " << mtx.row() << ", col : " << mtx.col() << "]";
    return os;
}

namespace sparseMatrix {
class DataBase {
 public:
  DataBase() = default;

  UIN row() const { return row_; }

  UIN col() const { return col_; }

  UIN nnz() const { return nnz_; }

  inline float getSparsity() const {
      return static_cast<float>(row_ * col_ - nnz_) / (row_ * col_);
  }

 protected:
  UIN row_;
  UIN col_;
  UIN nnz_;
};

template<typename T>
class CSR : public DataBase {
 public:
  CSR() = default;

  CSR(UIN row,
      UIN col,
      UIN nnz,
      const std::vector<UIN> &rowOffsets,
      const std::vector<UIN> &colIndices,
      const std::vector<T> &values) : rowOffsets_(rowOffsets), colIndices_(colIndices), values_(values) {
      row_ = row;
      col_ = col;
      nnz_ = nnz;
  }

  CSR(UIN row,
      UIN col,
      UIN nnz,
      const UIN *rowOffsets,
      const UIN *colIndices,
      const T *values) : rowOffsets_(rowOffsets, rowOffsets + row + 1),
                         colIndices_(colIndices, colIndices + nnz),
                         values_(values, values + nnz) {
      row_ = row;
      col_ = col;
      nnz_ = nnz;
  }

  CSR(UIN row,
      UIN col,
      UIN nnz,
      const int *rowOffsets,
      const int *colIndices,
      const T *values) : rowOffsets_(rowOffsets, rowOffsets + row + 1),
                         colIndices_(colIndices, colIndices + nnz),
                         values_(values, values + nnz) {
      row_ = row;
      col_ = col;
      nnz_ = nnz;
  }

  CSR(UIN row,
      UIN col,
      UIN nnz,
      const std::vector<UIN> &rowOffsets,
      const std::vector<UIN> &colIndices) : rowOffsets_(rowOffsets), colIndices_(colIndices), values_(nnz, 0) {
      row_ = row;
      col_ = col;
      nnz_ = nnz;
  }

  bool initializeFromMatrixFile(const std::string &file);

  /**
   * Initialize from smtx file.
   *
   * smtx file(CSR):
   *    1) The file begins with comment information beginning with %
   *    2) The storage format starts with a 3-integer header "nrows, ncols, nnz" that describes the number of rows in the matrix, the number of columns in the matrix, and the number of nonzeros in the matrix.
   *    3) The two rows represent rowOffset, and colIndex respectively
   **/
  bool initializeFromSmtxFile(const std::string &file);

  /**
   * Initialize from txt file.
   *
   * txt file(CSR):
   *
   **/
  bool initializeFromGraphDataset(const std::string &file);

  /**
  * Initialize from mtx file.
  *
  * mtx file format(COO):
  *    1) The file begins with comment information beginning with %
  *    2) By three numbers separated by a space: number of rows, number of columns, and number of non-zeros.
  *    3) Each after line has three numbers separated by a space: current row, current column, and value.
  **/
  bool initializeFromMtxFile(const std::string &file);

    /**
    * Used as a test comparison result
    **/
  bool outputToMarketMatrixFile(const std::string &fileName) const;

  bool outputToMarketMatrixFile() const;

  const std::vector<UIN> &rowOffsets() const { return rowOffsets_; }

  const std::vector<UIN> &colIndices() const { return colIndices_; }

  const std::vector<T> &values() const { return values_; }

  std::vector<T> &setValues() { return values_; }

 private:
  std::vector<UIN> rowOffsets_;
  std::vector<UIN> colIndices_;
  std::vector<T> values_;
};

template<typename T>
class COO : public DataBase {
 public:
  COO() = default;

  COO(UIN row,
      UIN col,
      UIN nnz,
      const std::vector<UIN> &rowIndices,
      const std::vector<UIN> &colIndices,
      const std::vector<T> &values) : rowIndices_(rowIndices), colIndices_(colIndices), values_(values) {
      row_ = row;
      col_ = col;
      nnz_ = nnz;
  }

  COO(const CSR<T> &csr);

  const std::vector<UIN> &rowIndices() const { return rowIndices_; }

  const std::vector<UIN> &colIndices() const { return colIndices_; }

  const std::vector<T> &values() const { return values_; }

  std::vector<T> &setValues() { return values_; }

  /**
   * Initialize from MatrixMarket file.
   *
   * MatrixMarket file format(COO):
   *    1) The file begins with comment information beginning with %
   *    2) By three numbers separated by a space: number of rows, number of columns, and number of non-zeros.
   *    3) Each after line has three numbers separated by a space: current row, current column, and value.
   **/
  bool initializeFromMatrixMarketFile(const std::string &file);

  /**
    * Used as a test comparison result
    **/
  bool outputToMarketMatrixFile(const std::string &fileName) const;

  bool outputToMarketMatrixFile() const;

  bool setValuesFromMatrix(const Matrix<T> &inputMatrix);

  void makeData(const UIN row, const UIN col, const UIN nnz);

  /**
   * input : idx
   * output : row, col, value
   **/
  std::tuple<UIN, UIN, T> getSpareMatrixOneData(const UIN idx) const;

  void draw() const;

  sparseMatrix::CSR<T> getCsrData() const;

  void print() const;

  std::tuple<UIN, UIN, T> operator[](UIN idx) const {
      if (idx > nnz_) {
          std::cerr << "Error! Array access out of bounds" << std::endl;
      }
      return std::make_tuple(rowIndices_[idx], colIndices_[idx], values_[idx]);
  }

 private:
  std::vector<UIN> rowIndices_;
  std::vector<UIN> colIndices_;
  std::vector<T> values_;
};

template<typename T>
class BELL : public DataBase {
 public:
  BELL() = default;

  BELL(UIN row,
       UIN col,
       UIN nnz,
       const std::vector<T> &blockRowOffsets,
       const std::vector<UIN> &blockColIndices,
       const std::vector<T> &blockValues)
      : blockRowOffsets_(blockRowOffsets), blockColIndices_(blockColIndices), blockValues_(blockValues) {
      row_ = row;
      col_ = col;
      nnz_ = nnz;
  }

  const std::vector<UIN> &blockRowOffsets() const { return blockRowOffsets_; }

  const std::vector<UIN> &blockColIndices() const { return blockColIndices_; }

  const std::vector<T> &blockValues() const { return blockValues_; }

 private:
  std::vector<UIN> blockRowOffsets_;
  std::vector<UIN> blockColIndices_;
  std::vector<T> blockValues_;
};
} // namespace sparseDataType

template<typename T>
bool checkMatrixData(const sparseMatrix::CSR<T> &csr);