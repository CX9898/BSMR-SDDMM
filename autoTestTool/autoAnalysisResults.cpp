#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>

const std::string dataSplitSymbol("---New data---");

struct ResultsInformation {
 public:

  void initInformation(const std::string &line);
  void clear();

  std::string checkData_;

  std::string gpu_;
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
  bool is_initialized_checkData_ = false;

  bool is_initialized_gpu_ = false;
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
    gpu_.clear();
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
    auto initOperation = [](const std::string &line, const std::string &find,
                            bool &is_initialized, std::string &output) -> void {
      if (!is_initialized) {
          if (contains(line, find)) {
              const size_t beginIdx = line.find(find) + find.size();
              size_t endIdx = beginIdx;
              while (line[endIdx++] != '@') {}
              output = line.substr(beginIdx, endIdx - beginIdx - 2);
              is_initialized = true;
          }
      }
    };

    auto initSettingOperation = [](const std::string &line, const std::string &find,
                                   bool &is_initialized, std::string &output) -> void {
      if (!is_initialized) {
          if (contains(line, find)) {
              const size_t beginIdx = line.find(find) + 1;
              size_t endIdx = beginIdx + 1;
              while (line[endIdx++] != '@') {}
              output = line.substr(beginIdx, endIdx - beginIdx - 2);
              is_initialized = true;
          }
      }
    };

    initSettingOperation(line, "@Device : ", is_initialized_gpu_, gpu_);
    initSettingOperation(line, "@WMMA_M : ", is_initialized_wmma_m_, wmma_m_);
    initSettingOperation(line, "@WMMA_N : ", is_initialized_wmma_n_, wmma_n_);
    initSettingOperation(line, "@WMMA_K : ", is_initialized_wmma_k_, wmma_k_);

    initSettingOperation(line, "@matrixA type : ", is_initialized_matrixA_type_, matrixA_type_);
    initSettingOperation(line, "@matrixB type : ", is_initialized_matrixB_type_, matrixB_type_);
    initSettingOperation(line, "@matrixC type : ", is_initialized_matrixC_type_, matrixC_type_);

    initSettingOperation(line, "@matrixA storageOrder : ", is_initialized_matrixA_storageOrder_, matrixA_storageOrder_);
    initSettingOperation(line, "@matrixB storageOrder : ", is_initialized_matrixB_storageOrder_, matrixB_storageOrder_);
    initSettingOperation(line, "@matrixC storageOrder : ", is_initialized_matrixC_storageOrder_, matrixC_storageOrder_);

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

    initOperation(line, "@checkData : ", is_initialized_checkData_, checkData_);
}

void printSettingInformation(const ResultsInformation &resultsInformation) {

    printf("\n");

    printf("- %s\n", resultsInformation.gpu_.c_str());

    printf("- %s, %s\n", resultsInformation.matrixA_type_.c_str(), resultsInformation.matrixA_storageOrder_.c_str());
    printf("- %s, %s\n", resultsInformation.matrixB_type_.c_str(), resultsInformation.matrixB_storageOrder_.c_str());
    printf("- %s, %s\n", resultsInformation.matrixC_type_.c_str(), resultsInformation.matrixC_storageOrder_.c_str());

    printf("- %s\n", resultsInformation.wmma_m_.c_str());
    printf("- %s\n", resultsInformation.wmma_n_.c_str());
    printf("- %s\n", resultsInformation.wmma_k_.c_str());

    printf("\n");
}

void printHeadOfList() {
    printf("\n");
    printf(
        "| | sparsity | k | isratnisa_sddmm | zcx_sddmm | isratnisa_other | zcx_other | isratnisa | zcx |\n");
    printf(
        "|-|----------|---|-----------------|-----------|-----------------|-----------|-----------|-----|\n");
}

void printOneLineOfList(const ResultsInformation &resultsInformation) {
    auto printOneInformation = [](const std::string &information) -> void {
      std::cout << information << "|";
    };
    printf("|");
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

    printf("%s", resultsInformation.checkData_.c_str());
    printf("\n");
}

size_t findNumK(const std::vector<ResultsInformation> &resultsInformation) {
    std::unordered_set<int> kSet;
    for (const auto &iter : resultsInformation) {
        kSet.insert(std::stoi(iter.K_));
    }
    return kSet.size();
}

void sortResultsInformation(std::vector<ResultsInformation> &resultsInformation) {
    printf("sortResultsInformation...\n");

    std::sort(resultsInformation.begin(), resultsInformation.end(),
              [&](ResultsInformation &a, ResultsInformation &b) {
                const float M_a = a.M_.empty() ? 0 : std::stof(a.M_);
                const float M_b = b.M_.empty() ? 0 : std::stof(b.M_);

                const float N_a = a.N_.empty() ? 0 : std::stof(a.N_);
                const float N_b = b.N_.empty() ? 0 : std::stof(b.N_);

                if (M_a < M_b) {
                    return true;
                } else {
                    return N_a < N_b;
                }
              });

    size_t compareM = 0, compareN = 0;
    size_t endIdxForLastSorted = 0;
    for (int idx = 0; idx < resultsInformation.size(); ++idx) {
        const size_t curM = std::stoll(resultsInformation[idx].M_);
        const size_t curN = std::stoll(resultsInformation[idx].N_);
        if (curM != compareM || curN != compareN) {
            std::sort(resultsInformation.data() + endIdxForLastSorted, resultsInformation.data() + idx,
                      [&](ResultsInformation &a, ResultsInformation &b) {
                        const float sparsity_a = a.sparsity_.empty() ? 0 : std::stof(a.sparsity_);
                        const float sparsity_b = b.sparsity_.empty() ? 0 : std::stof(b.sparsity_);
                        return sparsity_a < sparsity_b;
                      });
            endIdxForLastSorted = idx;
            compareM = curM;
            compareN = curN;
        }
    }

    std::sort(resultsInformation.data() + endIdxForLastSorted,
              resultsInformation.data() + resultsInformation.size(),
              [&](ResultsInformation &a, ResultsInformation &b) {
                const float sparsity_a = a.sparsity_.empty() ? 0 : std::stof(a.sparsity_);
                const float sparsity_b = b.sparsity_.empty() ? 0 : std::stof(b.sparsity_);
                return sparsity_a < sparsity_b;
              });

    const size_t numK = findNumK(resultsInformation);
    for (int idx = 0; idx < resultsInformation.size(); idx += numK) {
        std::sort(resultsInformation.data() + idx, resultsInformation.data() + idx + numK,
                  [&](ResultsInformation &a, ResultsInformation &b) {
                    const int K_a = a.K_.empty() ? 0 : std::stoi(a.K_);
                    const int K_b = b.K_.empty() ? 0 : std::stoi(b.K_);
                    return K_a < K_b;
                  });
    }
}

void readLogFile(const std::string &file, std::vector<ResultsInformation> &resultsInformation) {
    std::ifstream inFile;
    inFile.open(file, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, Results file cannot be opened : " << file << std::endl;
        return;
    }

    int testResultId = -1;
    std::string line; // Store the data for each line
    while (getline(inFile, line) && testResultId < static_cast<int>(resultsInformation.size())) {
        if (line == dataSplitSymbol) {
            ++testResultId;
            continue;
        }
        resultsInformation[testResultId].initInformation(line);
    }

    std::cout << "File read over : " << file << std::endl;
}

int getNumData(const std::string &file) {
    std::ifstream inFile;
    inFile.open(file, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, Results file cannot be opened : " << file << std::endl;
        return 0;
    }

    int numData = 0;
    std::string line; // Store the data for each line
    while (getline(inFile, line)) {
        if (line == dataSplitSymbol) {
            ++numData;
        }
    }

    printf("File \"%s\" number of data : %d\n", file.data(), numData);

    return numData;
}

void printInformationToMarkDown(const std::vector<ResultsInformation> &resultsInformation) {

    const size_t numK = findNumK(resultsInformation);
    const size_t numDataSets = resultsInformation.size() / numK;

    printSettingInformation(resultsInformation[0]);
    size_t compareM = 0, compareN = 0;
    for (int dataSetId = 0; dataSetId < numDataSets; ++dataSetId) {

        const size_t curM = std::stol(resultsInformation[dataSetId * numK].M_);
        const size_t curN = std::stol(resultsInformation[dataSetId * numK].N_);
        if (curM != compareM || curN != compareN) {
            printf(" matrix %ld %ld\n", curM, curN);
            compareM = curM;
            compareN = curN;
        }

        printHeadOfList();
        for (int resIdx = dataSetId * numK; resIdx < (dataSetId + 1) * numK && resIdx < resultsInformation.size();
             ++resIdx) {
            printOneLineOfList(resultsInformation[resIdx]);
        }
        printf("\n");
    }

}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Please enter two files.");
    }

    std::string inputFilePath1;
    std::string inputFilePath2;

    inputFilePath1 = argv[1];
    inputFilePath2 = argv[2];

    printf("start analyze the data...\n");

    int numData1 = getNumData(inputFilePath1);
    int numData2 = getNumData(inputFilePath2);

    if (numData1 != numData2) {
        fprintf(stderr, "Error! The two files do not have the same amount of data\n");
        return -1;
    }

    std::vector<ResultsInformation> resultsInformation(numData1);

    readLogFile(inputFilePath1, resultsInformation);
    readLogFile(inputFilePath2, resultsInformation);

    sortResultsInformation(resultsInformation);

    printf("Markdown format:\n");

    printInformationToMarkDown(resultsInformation);

    return 0;
}