#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>

//struct TensorCoreInformation {
// public:
//  std::string wmma_m_;
//  std::string wmma_n_;
//  std::string wmma_k_;
//
//  std::string matrixA_type_;
//  std::string matrixB_type_;
//  std::string matrixC_type_;
//
//  std::string matrixA_storageOrder_;
//  std::string matrixB_storageOrder_;
//  std::string matrixC_storageOrder_;
// private:
//};

struct ResultsInformation {
 public:
  std::string gpu_ = "4090";
  std::string buildType_ = "Release build";

  std::string wmma_m_;
  std::string wmma_n_;
  std::string wmma_k_;

  std::string matrixA_type_;
  std::string matrixB_type_;
  std::string matrixC_type_;

  std::string matrixA_storageOrder_;
  std::string matrixB_storageOrder_;
  std::string matrixC_storageOrder_;

  std::string M_;
  std::string N_;
  std::string K_;
  std::string NNZ_;
  std::string sparsity_;

  std::string isratnisa_sddmm_ = " "; // TODO
  std::string zcx_sddmm_;

  std::string isratnisa_other_ = " "; // TODO
  std::string zcx_other_;

  std::string isratnisa_ = " "; // TODO
  std::string zcx_;

  void initInformation(const std::string &line);
  void clear();

 private:
  bool is_initialized_gpu_ = true;
  bool is_initialized_buildType_ = true;

  bool is_initialized_wmma_m_ = false;
  bool is_initialized_wmma_n_ = false;
  bool is_initialized_wmma_k_ = false;

  bool is_initialized_matrixA_type_ = false;
  bool is_initialized_matrixB_type_ = false;
  bool is_initialized_matrixC_type_ = false;

  bool is_initialized_matrixA_storageOrder_ = false;
  bool is_initialized_matrixB_storageOrder_ = false;
  bool is_initialized_matrixC_storageOrder_ = false;

  bool is_initialized_M_ = false;
  bool is_initialized_N_ = false;
  bool is_initialized_K_ = false;
  bool is_initialized_NNZ_ = false;
  bool is_initialized_sparsity_ = false;

  bool is_initialized_isratnisa_sddmm_ = true; // TODO
  bool is_initialized_zcx_sddmm_ = false;
  bool is_initialized_isratnisa_other_ = true; // TODO
  bool is_initialized_zcx_other_ = false;
  bool is_initialized_isratnisa_ = true; // TODO
  bool is_initialized_zcx_ = false;
};

void ResultsInformation::clear() {
    gpu_ = "4090";
    buildType_ = "Release build";

    wmma_m_.clear();
    wmma_n_.clear();
    wmma_k_.clear();

    matrixA_type_.clear();
    matrixB_type_.clear();
    matrixC_type_.clear();

    matrixA_storageOrder_;
    matrixB_storageOrder_;
    matrixC_storageOrder_;

    M_.clear();
    N_.clear();
    K_.clear();
    NNZ_.clear();
    sparsity_.clear();

    isratnisa_sddmm_ = " "; // TODO
    zcx_sddmm_.clear();

    isratnisa_other_ = " "; // TODO
    zcx_other_.clear();

    isratnisa_ = " "; // TODO
    zcx_.clear();

    is_initialized_gpu_ = true;
    is_initialized_buildType_ = true;

    is_initialized_wmma_m_ = false;
    is_initialized_wmma_n_ = false;
    is_initialized_wmma_k_ = false;

    is_initialized_matrixA_type_ = false;
    is_initialized_matrixB_type_ = false;
    is_initialized_matrixC_type_ = false;

    is_initialized_matrixA_storageOrder_ = false;
    is_initialized_matrixB_storageOrder_ = false;
    is_initialized_matrixC_storageOrder_ = false;

    is_initialized_M_ = false;
    is_initialized_N_ = false;
    is_initialized_K_ = false;
    is_initialized_NNZ_ = false;
    is_initialized_sparsity_ = false;

    is_initialized_isratnisa_sddmm_ = true; // TODO
    is_initialized_zcx_sddmm_ = false;
    is_initialized_isratnisa_other_ = true; // TODO
    is_initialized_zcx_other_ = false;
    is_initialized_isratnisa_ = true; // TODO
    is_initialized_zcx_ = false;
}

inline bool contains(const std::string &str, const std::string &toFind) {
    return str.find(toFind) != std::string::npos;
}

void ResultsInformation::initInformation(const std::string &line) {
    auto initOperation = [&](const std::string &line, const std::string &find,
                             bool &is_initialized, std::string &output) -> void {
      if (!is_initialized) {
          if (contains(line, find)) {
              const int beginIdx = line.find(find) + find.size();
              int endIdx = beginIdx;
              while (line[endIdx++] != '@') {}
              output = line.substr(beginIdx, endIdx - beginIdx - 2);
              is_initialized = true;
          }
      }
    };

    initOperation(line, "@WMMA_M : ", is_initialized_wmma_m_, wmma_m_);
    initOperation(line, "@WMMA_N : ", is_initialized_wmma_n_, wmma_n_);
    initOperation(line, "@WMMA_K : ", is_initialized_wmma_k_, wmma_k_);

    initOperation(line, "@matrixA type : ", is_initialized_matrixA_type_, matrixA_type_);
    initOperation(line, "@matrixB type : ", is_initialized_matrixB_type_, matrixB_type_);
    initOperation(line, "@matrixC type : ", is_initialized_matrixC_type_, matrixC_type_);

    initOperation(line, "@matrixA storageOrder : ", is_initialized_matrixA_storageOrder_, matrixA_storageOrder_);
    initOperation(line, "@matrixB storageOrder : ", is_initialized_matrixB_storageOrder_, matrixB_storageOrder_);
    initOperation(line, "@matrixC storageOrder : ", is_initialized_matrixC_storageOrder_, matrixC_storageOrder_);

    initOperation(line, "@M : ", is_initialized_M_, M_);
    initOperation(line, "@N : ", is_initialized_N_, N_);
    initOperation(line, "@K : ", is_initialized_K_, K_);
    initOperation(line, "@NNZ : ", is_initialized_NNZ_, NNZ_);
    initOperation(line, "@sparsity : ", is_initialized_sparsity_, sparsity_);

    initOperation(line, "@isratnisa_sddmm : ", is_initialized_isratnisa_sddmm_, isratnisa_sddmm_);
    initOperation(line, "@zcx_sddmm : ", is_initialized_zcx_sddmm_, zcx_sddmm_);
    initOperation(line, "@isratnisa_other : ", is_initialized_isratnisa_other_, isratnisa_other_);
    initOperation(line, "@zcx_other : ", is_initialized_zcx_other_, zcx_other_);
    initOperation(line, "@isratnisa : ", is_initialized_isratnisa_, isratnisa_);
    initOperation(line, "@zcx : ", is_initialized_zcx_, zcx_);
}

void printHeadOfList() {

    printf(
        "|                            | sparsity | k    | isratnisa_sddmm | zcx_sddmm | isratnisa_other | zcx_other | isratnisa | zcx     |\n");
    printf(
        "|----------------------------|----------|------|-----------------|-----------|-----------------|-----------|-----------|---------|\n");
}

void printOneLineOfList(const ResultsInformation &resultsInformation) {
    auto printOneInformation = [&](const std::string &information) -> void {
      std::cout << information << " |";
    };
    printf("| ");
    std::string
        matrixFileName("matrix_" + resultsInformation.M_ + "_" + resultsInformation.N_ + "_" + resultsInformation.NNZ_);
    printOneInformation(matrixFileName);
    printOneInformation(resultsInformation.sparsity_);
    printOneInformation(resultsInformation.K_);
    printOneInformation(resultsInformation.isratnisa_sddmm_);
    printOneInformation(resultsInformation.zcx_sddmm_);
    printOneInformation(resultsInformation.isratnisa_other_);
    printOneInformation(resultsInformation.zcx_other_);
    printOneInformation(resultsInformation.isratnisa_);
    printOneInformation(resultsInformation.zcx_);

    printf("\n");
}

int main(int argc, char *argv[]) {
    printf("start analyze the data and print it\n");

    const std::string inputFilePath(argv[1]);
    std::ifstream inFile;
    inFile.open(inputFilePath, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, Results file cannot be opened : " << inputFilePath << std::endl;
        return -1;
    }

    ResultsInformation resultsInformation;

    printf("Markdown table : \n");
    printHeadOfList();

    std::string line; // Store the data for each line
    while (getline(inFile, line)) {
        if (line == "---next---") {
            printOneLineOfList(resultsInformation);
            resultsInformation.clear();
            continue;
        }
        resultsInformation.initInformation(line);
    }

}