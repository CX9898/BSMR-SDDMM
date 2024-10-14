#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

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

  void initInformation(const std::string &line);
  void clear();

  std::string gpu_ = "4090"; // TODO
  std::string buildType_ = "Release build"; // TODO

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

  std::string isratnisa_sddmm_;
  std::string zcx_sddmm_;

  std::string isratnisa_other_;
  std::string zcx_other_;

  std::string isratnisa_;
  std::string zcx_;

 private:
  bool is_initialized_gpu_ = true; // TODO
  bool is_initialized_buildType_ = true; // TODO

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

  bool is_initialized_isratnisa_sddmm_ = false;
  bool is_initialized_zcx_sddmm_ = false;
  bool is_initialized_isratnisa_other_ = false;
  bool is_initialized_zcx_other_ = false;
  bool is_initialized_isratnisa_ = false;
  bool is_initialized_zcx_ = false;
};

void ResultsInformation::clear() {
    gpu_ = "4090"; // TODO
    buildType_ = "Release build"; // TODO

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

    isratnisa_sddmm_.clear();
    zcx_sddmm_.clear();

    isratnisa_other_.clear();
    zcx_other_.clear();

    isratnisa_.clear();
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

    is_initialized_isratnisa_sddmm_ = false;
    is_initialized_zcx_sddmm_ = false;
    is_initialized_isratnisa_other_ = false;
    is_initialized_zcx_other_ = false;
    is_initialized_isratnisa_ = false;
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

void sortResultsInformation(std::vector<ResultsInformation> &resultsInformation) {
    printf("sortResultsInformation...\n");
    std::sort(resultsInformation.begin(), resultsInformation.end(),
              [&](ResultsInformation &a, ResultsInformation &b) {
                const float M_a = std::stof(a.M_);
                const float M_b = std::stof(b.M_);
                if (M_a > M_b) {
                    return true;
                }

                const float sparsity_a = std::stof(a.sparsity_);
                const float sparsity_b = std::stof(b.sparsity_);
                if (sparsity_a > sparsity_b) {
                    return false;
                } else {
                    return true;
                }
              });
}

int main(int argc, char *argv[]) {
    printf("start analyze the data and print it\n");

    std::string inputFilePath_zcx;
    std::string inputFilePath_isratnisa;
    int numTestResult = 0;
    if (argc < 2) {
        inputFilePath_zcx = argv[1];
    } else {
        numTestResult = std::atoi(argv[1]);
        inputFilePath_zcx = argv[2];
        inputFilePath_isratnisa = argv[3];
    }

    std::ifstream inFile_zcx;
    inFile_zcx.open(inputFilePath_zcx, std::ios::in); // open file
    if (!inFile_zcx.is_open()) {
        std::cerr << "Error, Results file cannot be opened : " << inputFilePath_zcx << std::endl;
        return -1;
    }

    std::ifstream inFile_isratnisa;
    inFile_isratnisa.open(inputFilePath_isratnisa, std::ios::in); // open file
    if (!inFile_isratnisa.is_open()) {
        std::cerr << "Error, Results file cannot be opened : " << inputFilePath_isratnisa << std::endl;
        return -1;
    }

    std::vector<ResultsInformation> resultsInformation(numTestResult);

    std::string line; // Store the data for each line
    int testResultId = 0;
    while (getline(inFile_zcx, line)) {
        if (line == "---next---") {
            ++testResultId;
            continue;
        }
        if (testResultId > resultsInformation.size()) {
            std::cerr << "numTestResult > input number" << std::endl;
            break;
        }
        resultsInformation[testResultId].initInformation(line);
    }

    testResultId = 0;
    while (getline(inFile_isratnisa, line)) {
        if (line == "---next---") {
            ++testResultId;
            continue;
        }
        if (testResultId > resultsInformation.size()) {
            std::cerr << "numTestResult > input number" << std::endl;
            break;
        }
        resultsInformation[testResultId].initInformation(line);
    }
    sortResultsInformation(resultsInformation);

    printf("Markdown table : \n");
    printHeadOfList();
    for (int resIdx = 0; resIdx < numTestResult; ++resIdx) {
        printOneLineOfList(resultsInformation[resIdx]);
    }
}