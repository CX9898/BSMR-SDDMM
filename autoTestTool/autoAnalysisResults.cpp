#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <map>

const std::string dataSplitSymbol("---New data---");

std::string findWord(const std::string &line, const std::string &word) {
    size_t findIdx = line.find(word);
    if (findIdx != std::string::npos) {
        const size_t beginIdx = findIdx + 1;
        size_t endIdx = beginIdx + 1;
        while (line[endIdx++] != ']') {}
        return line.substr(beginIdx, endIdx - beginIdx - 1);
    }
    return "";
}

std::string findWord(const std::vector<std::string> &multiLine, const std::string &word) {
    std::string value;
    for (const std::string &line : multiLine) {
        value = findWord(line, word);
        if (!value.empty()) {
            break;
        }
    }
    return value;
}

std::string getValue(const std::string &line, const std::string &word) {
    size_t findIdx = line.find(word);
    if (findIdx != std::string::npos) {
        const size_t beginIdx = line.find(word) + word.size();
        size_t endIdx = beginIdx;
        while (line[endIdx++] != ']') {}
        return line.substr(beginIdx, endIdx - beginIdx - 1);
    }
    return "";
}

std::string getValue(const std::vector<std::string> &multiLine, const std::string &word) {
    std::string value;
    for (const std::string &line : multiLine) {
        value = getValue(line, word);
        if (!value.empty()) {
            break;
        }
    }
    return value;
}

// Initialize variables and check if they are different
bool initOperationOrCheckIfDifferent(std::string &src, const std::string &data) {
    if (src.empty()) {
        src = data;
    } else {
        if (!data.empty() && src != data) {
            fprintf(stderr, "Error, the value is different. src : %s, data : %s\n", src.c_str(), data.c_str());
            return false;
        }
    }
    return true;
}

struct SettingInformation {
  bool initInformation(const std::vector<std::string> &oneTimeData);

  void printInformation() const;

  std::string buildType_;
  std::string device_;

  std::string wmma_m_;
  std::string wmma_n_;
  std::string wmma_k_;

  std::string blockDim_;

  std::string matrixA_type_;
  std::string matrixB_type_;
  std::string matrixC_type_;

  std::string matrixA_storageOrder_;
  std::string matrixB_storageOrder_;
  std::string matrixC_storageOrder_;
};

// init the setting information, if setting information is already initialized, and the new data is different, than return false
bool SettingInformation::initInformation(const std::vector<std::string> &oneTimeResults) {
    std::string buildType, device, wmma_m, wmma_n, wmma_k, blockDim, matrixA_type, matrixB_type, matrixC_type,
        matrixA_storageOrder, matrixB_storageOrder, matrixC_storageOrder;

    for (const std::string &line : oneTimeResults) {
        buildType = buildType.empty() ? findWord(line, "[Build type : ") : buildType;
        device = device.empty() ? findWord(line, "[Device : ") : device;
        wmma_m = wmma_m.empty() ? findWord(line, "[WMMA_M : ") : wmma_m;
        wmma_n = wmma_n.empty() ? findWord(line, "[WMMA_N : ") : wmma_n;
        wmma_k = wmma_k.empty() ? findWord(line, "[WMMA_K : ") : wmma_k;
        blockDim = blockDim.empty() ? findWord(line, "[blockDim : ") : blockDim;
        matrixA_type = matrixA_type.empty() ? findWord(line, "[matrixA type : ") : matrixA_type;
        matrixB_type = matrixB_type.empty() ? findWord(line, "[matrixB type : ") : matrixB_type;
        matrixC_type = matrixC_type.empty() ? findWord(line, "[matrixC type : ") : matrixC_type;
        matrixA_storageOrder = matrixA_storageOrder.empty() ? findWord(line,
                                                                       "[matrixA storageOrder : ") : matrixA_storageOrder;
        matrixB_storageOrder = matrixB_storageOrder.empty() ? findWord(line,
                                                                       "[matrixB storageOrder : ") : matrixB_storageOrder;
        matrixC_storageOrder = matrixC_storageOrder.empty() ? findWord(line,
                                                                       "[matrixC storageOrder : ") : matrixC_storageOrder;
    }

    if (!initOperationOrCheckIfDifferent(buildType_, buildType)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(device_, device)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(wmma_m_, wmma_m)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(wmma_n_, wmma_n)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(wmma_k_, wmma_k)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(blockDim_, blockDim)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixA_type_, matrixA_type)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixB_type_, matrixB_type)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixC_type_, matrixC_type)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixA_storageOrder_, matrixA_storageOrder)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixB_storageOrder_, matrixB_storageOrder)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixC_storageOrder_, matrixC_storageOrder)) {
        return false;
    }

    return true;
}

void SettingInformation::printInformation() const {
    auto printOneInformation = [](const std::string &information) -> void {
      if (!information.empty()) {
          printf("- %s\n", information.c_str());
      }
    };

    printf("\n");

    printOneInformation(buildType_);
    printOneInformation(device_);
    printOneInformation(wmma_m_);
    printOneInformation(wmma_n_);
    printOneInformation(wmma_k_);
    printOneInformation(blockDim_);
    printOneInformation(matrixA_type_);
    printOneInformation(matrixB_type_);
    printOneInformation(matrixC_type_);
    printOneInformation(matrixA_storageOrder_);
    printOneInformation(matrixB_storageOrder_);
    printOneInformation(matrixC_storageOrder_);

    printf("\n");
}

struct OneTimeData {
  void initInformation(const std::vector<std::string> &oneTimeResults);
  std::string isratnisa_sddmm_;
  std::string zcx_sddmm_;

  std::string isratnisa_other_;
  std::string zcx_other_;

  std::string isratnisa_;
  std::string zcx_;
  std::string cuSparse_;

  std::string checkResults_;
};

void OneTimeData::initInformation(const std::vector<std::string> &oneTimeResults) {
    for (const std::string &line : oneTimeResults) {
        isratnisa_sddmm_ = isratnisa_sddmm_.empty() ? getValue(line, "[isratnisa_sddmm : ") : isratnisa_sddmm_;
        zcx_sddmm_ = zcx_sddmm_.empty() ? getValue(line, "[zcx_sddmm : ") : zcx_sddmm_;
        isratnisa_other_ = isratnisa_other_.empty() ? getValue(line, "[isratnisa_other : ") : isratnisa_other_;
        zcx_other_ = zcx_other_.empty() ? getValue(line, "[zcx_other : ") : zcx_other_;
        isratnisa_ = isratnisa_.empty() ? getValue(line, "[isratnisa : ") : isratnisa_;
        zcx_ = zcx_.empty() ? getValue(line, "[zcx : ") : zcx_;
        cuSparse_ = cuSparse_.empty() ? getValue(line, "[cuSparse : ") : cuSparse_;
        checkResults_ = checkResults_.empty() ? getValue(line, "[checkResults : ") : checkResults_;
    }
}

struct ResultsInformation {
  bool initInformation(const std::vector<std::string> &oneTimeResults);

  void printInformation() const;

  std::string file_;
  std::string M_;
  std::string N_;
  std::string NNZ_;
  std::string sparsity_;

  std::map<int, OneTimeData> kToOneTimeData_;
};

bool ResultsInformation::initInformation(const std::vector<std::string> &oneTimeResults) {
    std::string file, M, N, NNZ, sparsity, K_str;
    for (const std::string &line : oneTimeResults) {
        file = file.empty() ? getValue(line, "[File : ") : file;
        M = M.empty() ? getValue(line, "[M : ") : M;
        N = N.empty() ? getValue(line, "[N : ") : N;
        NNZ = NNZ.empty() ? getValue(line, "[NNZ : ") : NNZ;
        sparsity = sparsity.empty() ? getValue(line, "[sparsity : ") : sparsity;
        K_str = K_str.empty() ? getValue(line, "[K : ") : K_str;
    }

    if (!initOperationOrCheckIfDifferent(file_, file)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(M_, M)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(N_, N)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(NNZ_, NNZ)) {
        return false;
    }
    if (!initOperationOrCheckIfDifferent(sparsity_, sparsity)) {
        return false;
    }

    int k = std::stoi(K_str);
    if (kToOneTimeData_.find(k) == kToOneTimeData_.end()) {
        kToOneTimeData_[k] = OneTimeData();
    }
    kToOneTimeData_[k].initInformation(oneTimeResults);

    return true;
}

void ResultsInformation::printInformation() const {

    printf(" file: %s, sparsity: %s\n", file_.c_str(), sparsity_.c_str());

    // print the head of the list
    printf("\n");
    printf("|");
    printf(" file |");
    printf(" M |");
    printf(" N |");
    printf(" sparsity |");
    printf(" K |");
    printf(" isratnisa_sddmm |");
    printf(" zcx_sddmm |");
    printf(" isratnisa_other |");
    printf(" zcx_other |");
    printf(" isratnisa |");
    printf(" zcx |");
    printf(" cuSparse |");
    printf("\n");

    // print the split line
    const int numColData = 12;
    printf("|");
    for (int i = 0; i < numColData; ++i) {
        printf("--|");
    }
    printf("\n");

    auto printOneInformation = [](const std::string &information) -> void {
      std::cout << information << "|";
    };

    // Print data line by line
    for (const auto &iter : kToOneTimeData_) {
        printf("|");
        printOneInformation(file_);
        printOneInformation(M_);
        printOneInformation(N_);
        printOneInformation(sparsity_);
        std::cout << iter.first << "|";
        printOneInformation(iter.second.isratnisa_sddmm_);
        printOneInformation(iter.second.zcx_sddmm_);
        printOneInformation(iter.second.isratnisa_other_);
        printOneInformation(iter.second.zcx_other_);
        printOneInformation(iter.second.isratnisa_);
        printOneInformation(iter.second.zcx_);
        printOneInformation(iter.second.cuSparse_);
        std::cout << iter.second.checkResults_;
        printf("\n");
    }

    printf("\n");
}

// return the data in the file
std::vector<std::vector<std::string>> readResultsFile(const std::string &resultsFile) {
    std::vector<std::vector<std::string>> allData;

    std::ifstream inFile;
    inFile.open(resultsFile, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, Results file cannot be opened : " << resultsFile << std::endl;
    }

    std::vector<std::string> oneTimeData;
    std::string line; // Store the data for each line
    while (getline(inFile, line)) {
        if (line == dataSplitSymbol) {
            allData.push_back(oneTimeData);
            oneTimeData.clear();
            continue;
        }
        oneTimeData.push_back(line);
    }
    if (!oneTimeData.empty()) {
        allData.push_back(oneTimeData);
    }

    return allData;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Please input the file.\n");
        return -1;
    }

    SettingInformation settingInformation;
    std::unordered_map<std::string, ResultsInformation> matrixFileToResultsInformationMap;
    for (int fileIdx = 1; fileIdx < argc; ++fileIdx) {
        const std::string resultsFile = argv[fileIdx];

        const std::vector<std::vector<std::string>> allData = readResultsFile(resultsFile);

        for (const std::vector<std::string> &oneTimeResults : allData) {
            if (!settingInformation.initInformation(oneTimeResults)) {
                return -1;
            }
            const std::string matrixFile = findWord(oneTimeResults, "[File : ");
            if (matrixFile.empty()) {
                continue;
            }
            if (matrixFileToResultsInformationMap.find(matrixFile) == matrixFileToResultsInformationMap.end()) {
                matrixFileToResultsInformationMap[matrixFile] = ResultsInformation();
            }
            if (!matrixFileToResultsInformationMap[matrixFile].initInformation(oneTimeResults)) {
                return -1;
            }
        }
    }

    settingInformation.printInformation();

    for (const auto &iter : matrixFileToResultsInformationMap) {
        iter.second.printInformation();
    }

    return 0;
}