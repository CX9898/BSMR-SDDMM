#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <utility>
#include <limits>
#include <optional>

const std::string dataSplitSymbol("---New data---");

std::string findWord(const std::string& line, const std::string& word){
    const size_t findIdx = line.find(word);
    if (findIdx != std::string::npos){
        const size_t beginIdx = findIdx + 1;
        size_t endIdx = beginIdx + 1;
        while (line[endIdx++] != ']'){
        }
        return line.substr(beginIdx, endIdx - beginIdx - 1);
    }
    return "";
}

std::string getValue(const std::string& line, const std::string& word){
    size_t findIdx = line.find(word);
    if (findIdx != std::string::npos){
        const size_t beginIdx = line.find(word) + word.size();
        size_t endIdx = beginIdx;
        while (line[endIdx++] != ']'){
        }
        return line.substr(beginIdx, endIdx - beginIdx - 1);
    }
    return "";
}


template <typename T>
std::optional<T> tryParse(const std::string& str);

template <>
std::optional<int> tryParse<int>(const std::string& str){
    if (str.empty()) return std::nullopt;
    try{
        return std::stoi(str);
    }
    catch (...){
        return std::nullopt;
    }
}

template <>
std::optional<float> tryParse<float>(const std::string& str){
    if (str.empty()) return std::nullopt;
    try{
        return std::stof(str);
    }
    catch (...){
        return std::nullopt;
    }
}

std::string getValue(const std::vector<std::string>& multiLine, const std::string& word){
    std::string value;
    for (const std::string& line : multiLine){
        value = getValue(line, word);
        if (!value.empty()){
            break;
        }
    }
    return value;
}

void printSeparator(const std::string& title = ""){
    printf("\n---%s---\n",
           title.c_str());
}

// Initialize variables and check if they are different
bool initOperationOrCheckIfDifferent(std::string& src, const std::string& data){
    if (src.empty()){
        src = data;
    }
    else{
        if (!data.empty() && src != data){
            fprintf(stderr, "Error, the value is different. src : %s, data : %s\n", src.c_str(), data.c_str());
            return false;
        }
    }
    return true;
}

struct SettingInformation{
    bool initInformation(const std::vector<std::string>& oneTimeData);

    void printInformation() const;

    std::string buildType_;
    std::string device_;

    std::string wmma_m_;
    std::string wmma_n_;
    std::string wmma_k_;

    std::string blockDim_dense_;
    std::string blockDim_sparse_;

    std::string matrixA_type_;
    std::string matrixB_type_;
    std::string matrixC_type_;

    std::string matrixA_storageOrder_;
    std::string matrixB_storageOrder_;
    std::string matrixC_storageOrder_;
};

// init the setting information, if setting information is already initialized, and the new data is different, than return false
bool SettingInformation::initInformation(const std::vector<std::string>& oneTimeData){
    std::string buildType, device, wmma_m, wmma_n, wmma_k, blockDim_dense, blockDim_sparse, matrixA_type, matrixB_type,
                matrixC_type, matrixA_storageOrder, matrixB_storageOrder, matrixC_storageOrder;

    for (const std::string& line : oneTimeData){
        buildType = buildType.empty() ? findWord(line, "[Build type : ") : buildType;
        device = device.empty() ? findWord(line, "[Device : ") : device;
        wmma_m = wmma_m.empty() ? findWord(line, "[WMMA_M : ") : wmma_m;
        wmma_n = wmma_n.empty() ? findWord(line, "[WMMA_N : ") : wmma_n;
        wmma_k = wmma_k.empty() ? findWord(line, "[WMMA_K : ") : wmma_k;
        blockDim_dense = blockDim_dense.empty() ? findWord(line, "[blockDim_dense : ") : blockDim_dense;
        blockDim_sparse = blockDim_sparse.empty() ? findWord(line, "[blockDim_sparse : ") : blockDim_sparse;
        matrixA_type = matrixA_type.empty() ? findWord(line, "[matrixA type : ") : matrixA_type;
        matrixB_type = matrixB_type.empty() ? findWord(line, "[matrixB type : ") : matrixB_type;
        matrixC_type = matrixC_type.empty() ? findWord(line, "[matrixC type : ") : matrixC_type;
        matrixA_storageOrder = matrixA_storageOrder.empty()
                                   ? findWord(line,
                                              "[matrixA storageOrder : ")
                                   : matrixA_storageOrder;
        matrixB_storageOrder = matrixB_storageOrder.empty()
                                   ? findWord(line,
                                              "[matrixB storageOrder : ")
                                   : matrixB_storageOrder;
        matrixC_storageOrder = matrixC_storageOrder.empty()
                                   ? findWord(line,
                                              "[matrixC storageOrder : ")
                                   : matrixC_storageOrder;
    }

    if (!initOperationOrCheckIfDifferent(buildType_, buildType)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(device_, device)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(wmma_m_, wmma_m)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(wmma_n_, wmma_n)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(wmma_k_, wmma_k)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(blockDim_dense_, blockDim_dense)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(blockDim_sparse_, blockDim_sparse)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixA_type_, matrixA_type)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixB_type_, matrixB_type)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixC_type_, matrixC_type)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixA_storageOrder_, matrixA_storageOrder)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixB_storageOrder_, matrixB_storageOrder)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(matrixC_storageOrder_, matrixC_storageOrder)){
        return false;
    }

    return true;
}

void SettingInformation::printInformation() const{
    auto printOneInformation = [](const std::string& information) -> void{
        if (!information.empty()){
            printf("- %s\n", information.c_str());
        }
    };

    printf("\n");

    printOneInformation(device_);
    printOneInformation(buildType_);
    printOneInformation(wmma_m_);
    printOneInformation(wmma_n_);
    printOneInformation(wmma_k_);
    printOneInformation(blockDim_dense_);
    printOneInformation(blockDim_sparse_);
    printOneInformation(matrixA_type_);
    printOneInformation(matrixB_type_);
    printOneInformation(matrixC_type_);
    printOneInformation(matrixA_storageOrder_);
    printOneInformation(matrixB_storageOrder_);
    printOneInformation(matrixC_storageOrder_);

    printf("\n");
}

class BaseLine{
public:
    BaseLine(const std::string& nameToken): nameToken_(nameToken){
    }

    void parseLine(const std::vector<std::string>& multiLine){
        for (const std::string& line : multiLine){
            parseLine(line);
        }
    }

    void parseLine(const std::string& line){
        updateNumDenseBlock(tryParse<int>(getValue(line, "[" + nameToken_ + "_numDenseBlock : ")).value_or(0));
        updateGflops(tryParse<float>(getValue(line, "[" + nameToken_ + "_gflops : ")).value_or(0.0f));
        updateReordering(tryParse<float>(getValue(line, "[" + nameToken_ + "_reordering : ")).value_or(0.0f));
        updateRowReordering(tryParse<float>(getValue(line, "[" + nameToken_ + "_rowReordering : ")).value_or(0.0f));
        updateColReordering(tryParse<float>(getValue(line, "[" + nameToken_ + "_colReordering : ")).value_or(0.0f));
        updateCheckResults(getValue(line, "[" + nameToken_ + "_checkResults : "));
    }

    float gflops() const{ return gflops_; }
    int numDenseBlock() const{ return numDenseBlock_; }
    const std::string& checkResults() const{ return checkResults_; }
    float reordering() const{ return reordering_; }
    float rowReordering() const{ return rowReordering_; }
    float colReordering() const{ return colReordering_; }

private:
    std::string nameToken_;
    float gflops_ = 0.0f;
    int numDenseBlock_ = 0;
    float reordering_ = std::numeric_limits<float>::max();
    float rowReordering_ = std::numeric_limits<float>::max();
    float colReordering_ = std::numeric_limits<float>::max();
    std::string checkResults_;

    void updateGflops(const float newValue){
        gflops_ = std::max(gflops_, newValue);
    }

    void updateNumDenseBlock(const int newValue){
        numDenseBlock_ = std::max(numDenseBlock_, newValue);
    }

    void updateReordering(const float newValue){
        if (newValue < 1e-6){
            return; // Ignore very small values
        }
        reordering_ = std::min(reordering_, newValue);
    }

    void updateRowReordering(const float newValue){
        if (newValue < 1e-6){
            return; // Ignore very small values
        }
        rowReordering_ = std::min(rowReordering_, newValue);
    }

    void updateColReordering(const float newValue){
        if (newValue < 1e-6){
            return; // Ignore very small values
        }
        colReordering_ = std::min(colReordering_, newValue);
    }

    void updateCheckResults(const std::string& newValue){
        if (newValue.empty()){
            return; // Ignore empty results
        }
        checkResults_ = newValue;
    }
};

struct OneTimeData{
    void update(const std::vector<std::string>& oneTimeResults);

    BaseLine BSMR_{"bsmr"};
    BaseLine cuSDDMM_{"cuSDDMM"};
    BaseLine cuSparse_{"cuSparse"};
    BaseLine ASpT_{"ASpT"};
    BaseLine RoDe_{"RoDe"};
    BaseLine BSA_{"BSA"};
};

void OneTimeData::update(const std::vector<std::string>& oneTimeResults){
    BSMR_.parseLine(oneTimeResults);
    cuSDDMM_.parseLine(oneTimeResults);
    cuSparse_.parseLine(oneTimeResults);
    ASpT_.parseLine(oneTimeResults);
    RoDe_.parseLine(oneTimeResults);
    BSA_.parseLine(oneTimeResults);
}

struct ResultsInformation{
    bool initInformation(const std::vector<std::string>& oneTimeResults);

    void printInformation() const;

    bool empty() const{
        return kToOneTimeData_.empty();
    }

    std::string file_;
    std::string M_;
    std::string N_;
    std::string NNZ_;
    std::string sparsity_;

    std::map<int, OneTimeData> kToOneTimeData_;
};

bool ResultsInformation::initInformation(const std::vector<std::string>& oneTimeResults){
    std::string file, M, N, NNZ, sparsity, K_str;
    for (const std::string& line : oneTimeResults){
        file = file.empty() ? getValue(line, "[File : ") : file;
        M = M.empty() ? getValue(line, "[M : ") : M;
        N = N.empty() ? getValue(line, "[N : ") : N;
        NNZ = NNZ.empty() ? getValue(line, "[NNZ : ") : NNZ;
        sparsity = sparsity.empty() ? getValue(line, "[sparsity : ") : sparsity;
        K_str = K_str.empty() ? getValue(line, "[K : ") : K_str;
    }

    if (!initOperationOrCheckIfDifferent(file_, file)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(M_, M)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(N_, N)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(NNZ_, NNZ)){
        return false;
    }
    if (!initOperationOrCheckIfDifferent(sparsity_, sparsity)){
        return false;
    }

    int k = std::stoi(K_str);
    if (kToOneTimeData_.find(k) == kToOneTimeData_.end()){
        kToOneTimeData_[k] = OneTimeData();
    }
    kToOneTimeData_[k].update(oneTimeResults);

    return true;
}

std::string getFileName(const std::string& path){
    if (path.empty()) return "";

    const size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos){
        std::cerr << "Warning. The input path has no parent folder" << std::endl;
    }
    const std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
    return filename;
}

void ResultsInformation::printInformation() const{
    printf("## M : %s, N: %s, sparsity: %s, file: %s\n",
           M_.c_str(), N_.c_str(), sparsity_.c_str(), file_.c_str());

    const int numColAttributes = 10;

    // print the head of the list
    printf("\n");
    printf("|");
    printf(" M |");
    printf(" N |");
    printf(" NNZ |");
    printf(" sparsity |");
    printf(" K |");
    printf(" bsmr_gflops |");
    printf(" cuSDDMM_gflops |");
    printf(" cuSparse_gflops |");
    printf(" ASpT_gflops |");
    printf(" RoDe_gflops |");

    printf("\n");

    // print the split line
    const int numColData = numColAttributes;
    printf("|");
    for (int i = 0; i < numColData; ++i){
        printf("-|");
    }
    printf("\n");

    auto printOneLineInformation = [](const std::string& information) -> void{
        std::cout << information << "|";
    };

    // Print data line by line
    for (const auto& iter : kToOneTimeData_){
        printf("|");
        printOneLineInformation(M_);
        printOneLineInformation(N_);
        printOneLineInformation(NNZ_);
        printOneLineInformation(sparsity_);
        std::cout << iter.first << "|"; // K value
        std::cout << iter.second.BSMR_.gflops() << "|";
        std::cout << iter.second.cuSDDMM_.gflops() << "|";
        std::cout << iter.second.cuSparse_.gflops() << "|";
        std::cout << iter.second.ASpT_.gflops() << "|";
        std::cout << iter.second.RoDe_.gflops() << "|";

        std::cout << iter.second.BSMR_.checkResults();
        printf("\n");
    }

    printf("\n");
}

// return the data in the file
std::vector<std::vector<std::string>> readResultsFile(const std::string& resultsFile){
    std::vector<std::vector<std::string>> allData;

    std::ifstream inFile;
    inFile.open(resultsFile, std::ios::in); // open file
    if (!inFile.is_open()){
        std::cerr << "Error, results file cannot be opened : " << resultsFile << std::endl;
    }

    std::vector<std::string> oneTimeData;
    std::string line; // Store the data for each line
    while (getline(inFile, line)){
        if (line == dataSplitSymbol){
            allData.push_back(oneTimeData);
            oneTimeData.clear();
            continue;
        }
        oneTimeData.push_back(line);
    }
    if (!oneTimeData.empty()){
        allData.push_back(oneTimeData);
    }

    return allData;
}

std::unordered_map<std::string, ResultsInformation> pickTheBadResults(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    std::unordered_map<std::string, ResultsInformation> bad;

    for (const auto& iter : matrixFileToResultsInformationMap){
        const std::string file = iter.first;
        const ResultsInformation& resultesInformation = iter.second;

        ResultsInformation badResultsInformation(resultesInformation);
        badResultsInformation.kToOneTimeData_.clear();
        for (const auto& kToOneTimeData : resultesInformation.kToOneTimeData_){
            const int k = kToOneTimeData.first;
            const float bsmr_gflops = kToOneTimeData.second.BSMR_.gflops();
            const float cuSDDMM_gflops = kToOneTimeData.second.cuSDDMM_.gflops();
            const float cuSparse_gflops = kToOneTimeData.second.cuSparse_.gflops();
            if (bsmr_gflops > 1e-6){
                if (bsmr_gflops < cuSDDMM_gflops || bsmr_gflops < cuSparse_gflops){
                    OneTimeData oneTimeData = kToOneTimeData.second;
                    badResultsInformation.kToOneTimeData_[k] = oneTimeData;
                }
            }

            const float ASpT_gflops = kToOneTimeData.second.ASpT_.gflops();
            if (bsmr_gflops > 1e-6){
                if (bsmr_gflops < ASpT_gflops){
                    OneTimeData oneTimeData = kToOneTimeData.second;
                    badResultsInformation.kToOneTimeData_[k] = oneTimeData;
                }
            }
        }
        if (!badResultsInformation.empty()){
            bad[file] = badResultsInformation;
        }
    }

    return bad;
}

int getNumResults(const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        numResults += iter.second.kToOneTimeData_.size();

        for (const auto& kToOneTimeData : iter.second.kToOneTimeData_){
            const float bsmr_sddmm = kToOneTimeData.second.BSMR_.gflops();

            if (bsmr_sddmm <= 1e-6){
                --numResults;
            }
        }
    }

    return numResults;
}

bool checkIsCorrect(const std::string& checkResults){
    std::string str = checkResults;
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);

    const bool isNoPass = checkResults.find("no pass");

    if (isNoPass){
        float errorRate = 0.0f;
        const size_t beginIdx = str.find("error rate : ");
        const size_t endIdx = str.find('%');
        if (beginIdx != std::string::npos && endIdx != std::string::npos){
            errorRate = std::stof(str.substr(beginIdx + 13, endIdx - beginIdx - 13));
        }

        if (errorRate > 1e-6){
            return false;
        }
    }

    return true;
}

float calculateAccuracy(const std::unordered_map<std::string,
                                                 ResultsInformation>& matrixFileToResultsInformationMap){
    const int numResults = getNumResults(matrixFileToResultsInformationMap);
    int numErrors = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        for (const auto& kToOneTimeData : iter.second.kToOneTimeData_){
            bool isCorrect = checkIsCorrect(kToOneTimeData.second.BSMR_.checkResults());

            if (!isCorrect){
                ++numErrors;
            }
        }
    }
    const float accuracy = 1.0f - static_cast<float>(numErrors) / numResults;

    printf("Accuracy: %.2f%%\n", accuracy * 100);

    return accuracy;
}

// return the average speedup adn the maximum speedup
void evaluateSddmmWithCuSDDMM(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // 0~0.5, 0.5~0.8, 0.8~1.0, 1.0~1.2, 1.2~1.5, 1.5~
    std::vector<int> numSpeedups(6);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        for (const auto& kToOneTimeData : iter.second.kToOneTimeData_){
            const float bsmr_gflops = kToOneTimeData.second.BSMR_.gflops();
            const float cuSDDMM_gflops = kToOneTimeData.second.cuSDDMM_.gflops();

            if (bsmr_gflops <= 1e-6 || cuSDDMM_gflops <= 1e-6){
                continue;
            }

            float speedup = bsmr_gflops / cuSDDMM_gflops;
            maxSpeedup = std::max(speedup, maxSpeedup);
            sumSpeedup += speedup;
            ++numResults;

            if (speedup <= 0.5){
                ++numSpeedups[0];
            }
            if (speedup > 0.5 && speedup <= 0.8){
                ++numSpeedups[1];
            }
            if (speedup > 0.8 && speedup <= 1.0){
                ++numSpeedups[2];
            }
            if (speedup > 1.0 && speedup <= 1.2){
                ++numSpeedups[3];
            }
            if (speedup > 1.2 && speedup <= 1.5){
                ++numSpeedups[4];
            }
            if (speedup > 1.5){
                ++numSpeedups[5];
            }
        }
    }

    float averageSpeedup = sumSpeedup / numResults;

    printSeparator("Evaluate sddmm With cuSDDMM:");
    printf("Number of results: %d\n", numResults);
    printf("Average speedup over cuSDDMM: %.2fx, maximum speedup: %.2fx\n", averageSpeedup, maxSpeedup);

    printf("Speedup over cuSDDMM <= 0.5 : %.1f%%\n", numSpeedups[0] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over cuSDDMM 0.5~0.8 : %.1f%%\n", numSpeedups[1] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over cuSDDMM 0.8~1.0 : %.1f%%\n", numSpeedups[2] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over cuSDDMM 1.0~1.2 : %.1f%%\n", numSpeedups[3] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over cuSDDMM 1.2~1.5 : %.1f%%\n", numSpeedups[4] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over cuSDDMM > 1.5 : %.1f%%\n", numSpeedups[5] / static_cast<float>(numResults) * 100.0f);

    const float accelerationCoverage =
        (numSpeedups[3] + numSpeedups[4] + numSpeedups[5]) / static_cast<float>(numResults) * 100.0f;
    printf("Acceleration coverage: %.1f%%\n", accelerationCoverage);

    printf("\n");
}

// return the average speedup adn the maximum speedup
void evaluateSddmmWithCuSparse(
    std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        for (const auto& kToOneTimeData : iter.second.kToOneTimeData_){
            const float bsmr_sddmm = kToOneTimeData.second.BSMR_.gflops();
            const float cuSparse_sddmm = kToOneTimeData.second.cuSparse_.gflops();

            if (bsmr_sddmm <= 1e-6 || cuSparse_sddmm <= 1e-6){
                continue;
            }

            float speedup = bsmr_sddmm / cuSparse_sddmm;
            maxSpeedup = std::max(speedup, maxSpeedup);
            sumSpeedup += speedup;
            ++numResults;
        }
    }

    float averageSpeedup = sumSpeedup / numResults;

    printSeparator("evaluateSddmmWithCuSparse:");
    printf("Number of results: %d\n", numResults);
    printf("Average speedup over cuSparse: %.2f, maximum speedup: %.2f\n", averageSpeedup, maxSpeedup);

    printf("\n");
}

// return the average speedup adn the maximum speedup
void evaluateSddmmWithASpT(
    std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // 0~0.5, 0.5~0.8, 0.8~1.0, 1.0~1.2, 1.2~1.5, 1.5~
    std::vector<int> numSpeedups(6);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        for (const auto& kToOneTimeData : iter.second.kToOneTimeData_){
            const float bsmr_sddmm = kToOneTimeData.second.BSMR_.gflops();
            const float ASpT_sddmm = kToOneTimeData.second.ASpT_.gflops();

            if (bsmr_sddmm <= 1e-6 || ASpT_sddmm <= 1e-6){
                continue;
            }

            const float speedup = bsmr_sddmm / ASpT_sddmm;
            maxSpeedup = std::max(speedup, maxSpeedup);
            sumSpeedup += speedup;
            ++numResults;

            if (speedup <= 0.5){
                ++numSpeedups[0];
            }
            if (speedup > 0.5 && speedup <= 0.8){
                ++numSpeedups[1];
            }
            if (speedup > 0.8 && speedup <= 1.0){
                ++numSpeedups[2];
            }
            if (speedup > 1.0 && speedup <= 1.2){
                ++numSpeedups[3];
            }
            if (speedup > 1.2 && speedup <= 1.5){
                ++numSpeedups[4];
            }
            if (speedup > 1.5){
                ++numSpeedups[5];
            }
        }
    }

    float averageSpeedup = sumSpeedup / numResults;

    printSeparator("Evaluate sddmm With ASpT:");
    printf("Number of results: %d\n", numResults);
    printf("Average speedup over ASpT: %.2fx, maximum speedup: %.2fx\n", averageSpeedup, maxSpeedup);

    printf("Speedup over ASpT <= 0.5 : %.1f%%\n", numSpeedups[0] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over ASpT 0.5~0.8 : %.1f%%\n", numSpeedups[1] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over ASpT 0.8~1.0 : %.1f%%\n", numSpeedups[2] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over ASpT 1.0~1.2 : %.1f%%\n", numSpeedups[3] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over ASpT 1.2~1.5 : %.1f%%\n", numSpeedups[4] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over ASpT > 1.5 : %.1f%%\n", numSpeedups[5] / static_cast<float>(numResults) * 100.0f);

    const float accelerationCoverage =
        (numSpeedups[3] + numSpeedups[4] + numSpeedups[5]) / static_cast<float>(numResults) * 100.0f;
    printf("Acceleration coverage: %.1f%%\n", accelerationCoverage);

    printf("\n");
}

// return the average speedup adn the maximum speedup
void evaluateSddmmWithRoDe(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // 0~0.5, 0.5~0.8, 0.8~1.0, 1.0~1.2, 1.2~1.5, 1.5~
    std::vector<int> numSpeedups(6);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        for (const auto& kToOneTimeData : iter.second.kToOneTimeData_){
            const float bsmr_sddmm = kToOneTimeData.second.BSMR_.gflops();
            const float RoDe_sddmm = kToOneTimeData.second.RoDe_.gflops();

            if (bsmr_sddmm <= 1e-6 || RoDe_sddmm <= 1e-6){
                continue;
            }

            const float speedup = bsmr_sddmm / RoDe_sddmm;
            maxSpeedup = std::max(speedup, maxSpeedup);
            sumSpeedup += speedup;
            ++numResults;

            if (speedup <= 0.5){
                ++numSpeedups[0];
            }
            if (speedup > 0.5 && speedup <= 0.8){
                ++numSpeedups[1];
            }
            if (speedup > 0.8 && speedup <= 1.0){
                ++numSpeedups[2];
            }
            if (speedup > 1.0 && speedup <= 1.2){
                ++numSpeedups[3];
            }
            if (speedup > 1.2 && speedup <= 1.5){
                ++numSpeedups[4];
            }
            if (speedup > 1.5){
                ++numSpeedups[5];
            }
        }
    }

    float averageSpeedup = sumSpeedup / numResults;

    printSeparator("Evaluate sddmm With RoDe:");
    printf("Number of results: %d\n", numResults);
    printf("Average speedup over RoDe: %.2fx, maximum speedup: %.2fx\n", averageSpeedup, maxSpeedup);

    printf("Speedup over RoDe <= 0.5 : %.1f%%\n", numSpeedups[0] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over RoDe 0.5~0.8 : %.1f%%\n", numSpeedups[1] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over RoDe 0.8~1.0 : %.1f%%\n", numSpeedups[2] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over RoDe 1.0~1.2 : %.1f%%\n", numSpeedups[3] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over RoDe 1.2~1.5 : %.1f%%\n", numSpeedups[4] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup over RoDe > 1.5 : %.1f%%\n", numSpeedups[5] / static_cast<float>(numResults) * 100.0f);

    const float accelerationCoverage =
        (numSpeedups[3] + numSpeedups[4] + numSpeedups[5]) / static_cast<float>(numResults) * 100.0f;
    printf("Acceleration coverage: %.1f%%\n", accelerationCoverage);

    printf("\n");
}

void eliminateInvalidData(std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    for (auto iter = matrixFileToResultsInformationMap.begin(); iter != matrixFileToResultsInformationMap.end();){
        const int m = tryParse<int>(iter->second.M_).value_or(0);
        const int n = tryParse<int>(iter->second.N_).value_or(0);
        if (m < 5000 || n < 5000){
            iter = matrixFileToResultsInformationMap.erase(iter);
        }
        else{
            ++iter;
        }
    }
}

void printReorderingEffectiveness(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    printSeparator("Reordering Effectiveness:");

    const int numColAttributes = 3;
    // print the head of the list
    printf("\n");
    printf("|");
    printf(" NNZ |");
    printf(" bsmr_numDenseBlock |");
    printf(" bsa_numDenseBlock |");

    printf("\n");

    // print the split line
    const int numColData = numColAttributes;
    printf("|");
    for (int i = 0; i < numColData; ++i){
        printf("-|");
    }
    printf("\n");

    for (const auto& iter : matrixFileToResultsInformationMap){
        for (const auto& kToOneTimeData : iter.second.kToOneTimeData_){
            const int bsmr_numDenseBlock = kToOneTimeData.second.BSMR_.numDenseBlock();
            const int bsa_numDenseBlock = kToOneTimeData.second.BSA_.numDenseBlock();

            if (bsmr_numDenseBlock <= 0 && bsa_numDenseBlock <= 0){
                continue;
            }

            printf("|");
            std::cout << iter.second.NNZ_ << "|";
            std::cout << bsmr_numDenseBlock << "|";
            std::cout << bsa_numDenseBlock << "|";
            printf("\n");
        }
    }

    printf("\n");
    printf("\n");
}

void evaluateReorderingOverhead(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    printSeparator("Evaluate Reordering Overhead:");

    const int numColAttributes = 4;
    // print the head of the list
    printf("\n");
    printf("|");
    printf(" rows |");
    printf(" cols |");
    printf(" rowReordering |");
    printf(" colReordering |");

    printf("\n");

    // print the split line
    const int numColData = numColAttributes;
    printf("|");
    for (int i = 0; i < numColData; ++i){
        printf("-|");
    }
    printf("\n");

    for (const auto& iter : matrixFileToResultsInformationMap){
        const int rows = std::stoi(iter.second.M_);
        const int cols = std::stoi(iter.second.N_);
        const float reorderingTime = iter.second.kToOneTimeData_.begin()->second.BSMR_.reordering();
        const float rowReorderingTime = iter.second.kToOneTimeData_.begin()->second.BSMR_.rowReordering();
        const float colReorderingTime = iter.second.kToOneTimeData_.begin()->second.BSMR_.colReordering();
        if (reorderingTime == std::numeric_limits<float>::max()){
            continue;
        }

        printf("|");
        std::cout << rows << "|";
        std::cout << cols << "|";
        std::cout << rowReorderingTime << "|";
        std::cout << colReorderingTime << "|";
        printf("\n");
    }

    printf("\n");
}

void evaluateReorderingWithBSA(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    int sumZCX = 0;
    int sumBSA = 0;

    for (const auto& iter : matrixFileToResultsInformationMap){
        for (const auto& kToOneTimeData : iter.second.kToOneTimeData_){
            // if (kToOneTimeData.second.bsmr_numDenseBlock_.empty() || kToOneTimeData.second.BSA_numDenseBlock_.empty()) {
            //     continue;
            // }

            const int bsmr_numDenseBlock = kToOneTimeData.second.BSMR_.numDenseBlock();
            const int bsa_numDenseBlock = kToOneTimeData.second.BSA_.numDenseBlock();

            sumZCX += bsmr_numDenseBlock;
            sumBSA += bsa_numDenseBlock;
        }
    }

    const float relativeIncreasePercent = static_cast<float>(sumZCX - sumBSA) / sumBSA * 100.0f;

    printSeparator("Evaluate Reordering With BSA:");

    printf("Percent increase in the number of dense blocks generated relative to BSA: %.2f%%\n",
           relativeIncreasePercent);

    printReorderingEffectiveness(matrixFileToResultsInformationMap);

    printf("\n");
}

void analyzeDataset(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    printSeparator("Dataset Analysis:");

    printf("Number of matrix files: %d\n", static_cast<int>(matrixFileToResultsInformationMap.size()));

    // Print the results Analysis information
    const int numResults = getNumResults(matrixFileToResultsInformationMap);
    printf("Number of data: %d\n", numResults);

    float maxSparsity = 0.0f;
    float minSparsity = 100.0f;

    int maxM = 0;
    int minM = std::numeric_limits<int>::max();;

    int maxN = 0;
    int minN = std::numeric_limits<int>::max();;

    for (const auto& iter : matrixFileToResultsInformationMap){
        const int M = tryParse<int>(iter.second.M_).value_or(0);
        const int N = tryParse<int>(iter.second.N_).value_or(0);
        const float sparsity = tryParse<float>(iter.second.sparsity_).value_or(0.0f);

        maxM = std::max(M, maxM);
        minM = std::min(M, minM);

        maxN = std::max(N, maxN);
        minN = std::min(N, minN);

        maxSparsity = std::max(sparsity, maxSparsity);
        minSparsity = std::min(sparsity, minSparsity);
    }

    printf("Maximum sparsity: %.2f%%, minimum sparsity: %.2f%%\n", maxSparsity, minSparsity);

    printf("Maximum m: %d, minimum m: %d\n", maxM, minM);
    printf("Maximum n: %d, minimum n: %d\n", maxN, minN);

    printf("\n");
}

int main(const int argc, const char* argv[]){
    // Read the results file
    SettingInformation settingInformation;
    std::unordered_map<std::string, ResultsInformation> matrixFileToResultsInformationMap;
    for (int fileIdx = 1; fileIdx < argc; ++fileIdx){
        const std::string resultsFile = argv[fileIdx];

        const std::vector<std::vector<std::string>> allData = readResultsFile(resultsFile);

        for (const std::vector<std::string>& oneTimeResults : allData){
            if (!settingInformation.initInformation(oneTimeResults)){
                return -1;
            }
            const std::string matrixFile = getValue(oneTimeResults, "[File : ");
            if (matrixFile.empty()){
                continue;
            }
            if (matrixFileToResultsInformationMap.find(matrixFile) == matrixFileToResultsInformationMap.end()){
                matrixFileToResultsInformationMap[matrixFile] = ResultsInformation();
            }
            if (!matrixFileToResultsInformationMap[matrixFile].initInformation(oneTimeResults)){
                return -1;
            }
        }
    }
    eliminateInvalidData(matrixFileToResultsInformationMap);

    analyzeDataset(matrixFileToResultsInformationMap);

    calculateAccuracy(matrixFileToResultsInformationMap);

    evaluateSddmmWithCuSparse(matrixFileToResultsInformationMap);

    evaluateSddmmWithCuSDDMM(matrixFileToResultsInformationMap);

    evaluateSddmmWithASpT(matrixFileToResultsInformationMap);

    evaluateSddmmWithRoDe(matrixFileToResultsInformationMap);

    // evaluateReorderingWithBSA(matrixFileToResultsInformationMap);

    // evaluateReorderingOverhead(matrixFileToResultsInformationMap);

    // Print the program setting information to Markdown format and the results information
    // settingInformation.printInformation();
    // for (const auto& iter : matrixFileToResultsInformationMap){
    //     iter.second.printInformation();
    // }

    return 0;
}
