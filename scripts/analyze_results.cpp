#include <cstdio>
#include <cmath>
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
#include <tuple>

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
    printf("\n---%s---\n", title.c_str());
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
    BaseLine() = default;

    BaseLine(const std::string& nameToken): nameToken_(nameToken){
    }

    virtual void parse(const std::vector<std::string>& oneTimeInformation){
        int K = 0;
        float gflops = 0.0f;
        for (const std::string& line : oneTimeInformation){
            K = std::max(K, tryParse<int>(getValue(line, "[K : ")).value_or(0));
            gflops = std::max(gflops,
                              tryParse<float>(getValue(line, "[" + nameToken_ + "_gflops : ")).value_or(0.0f));
        }
        if (gflops > gflops_){
            K_ = K;
            gflops_ = gflops;
        }
    }

    int K() const{ return K_; }
    float gflops() const{ return gflops_; }

protected:
    std::string nameToken_;

    int K_ = 0;
    float gflops_ = 0.0f;
};

// alpha -> delta -> (gflops, numDenseBlock, averageDensity, original_numDenseBlock, original_averageDensity, rowReorderingTime, colReorderingTime)
struct ReorderingData{
    float gflops = 0.0f;
    int numClusters = 0;
    int numDenseBlock = 0;
    float averageDensity = 0.0f;
    int original_numDenseBlock = 0;
    float original_averageDensity = 0.0f;
    float rowReorderingTime = 0.0f;
    float colReorderingTime = 0.0f;
};

class BaseLineWithParameter : public BaseLine{
public:
    BaseLineWithParameter(const std::string& nameToken): nameToken_(nameToken){
    }

    void parse(const std::vector<std::string>& oneTimeInformation){
        int K = 0;
        float gflops = 0.0f;
        float alpha = 0.0f;
        float delta = 0.0f;
        int numClusters = 0;
        int numDenseBlock = 0;
        float reordering = 0.0f;
        float rowReordering = 0.0f;
        float colReordering = 0.0f;
        std::string checkResults;
        float averageDensity = 0.0f;
        int original_numDenseBlock = 0;
        float original_averageDensity = 0.0f;
        for (const std::string& line : oneTimeInformation){
            K = std::max(K, tryParse<int>(getValue(line, "[K : ")).value_or(0));
            gflops = std::max(gflops,
                              tryParse<float>(getValue(line, "[" + nameToken_ + "_gflops : ")).value_or(0.0f));
            alpha = std::max(alpha,
                             tryParse<float>(getValue(line, "[" + nameToken_ + "_alpha : ")).value_or(0.0f));
            delta = std::max(delta,
                             tryParse<float>(getValue(line, "[" + nameToken_ + "_delta : ")).value_or(0.0f));
            numClusters = std::max(numClusters,
                                   tryParse<int>(getValue(line, "[" + nameToken_ + "_numClusters : ")).value_or(0));
            numDenseBlock = std::max(numDenseBlock,
                                     tryParse<int>(getValue(line, "[" + nameToken_ + "_numDenseBlock : ")).value_or(0));
            reordering = std::max(reordering,
                                  tryParse<float>(getValue(line, "[" + nameToken_ + "_reordering : ")).value_or(0.0f));
            rowReordering = std::max(rowReordering,
                                     tryParse<float>(getValue(line, "[" + nameToken_ + "_rowReordering : ")).value_or(
                                         0.0f));
            colReordering = std::max(colReordering,
                                     tryParse<float>(getValue(line, "[" + nameToken_ + "_colReordering : ")).value_or(
                                         0.0f));
            averageDensity = std::max(
                averageDensity,
                tryParse<float>(getValue(line, "[" + nameToken_ + "_averageDensity : ")).value_or(0.0f));
            original_numDenseBlock = std::max(
                original_numDenseBlock,
                tryParse<int>(getValue(line, "[original_numDenseBlock : ")).value_or(0));
            original_averageDensity = std::max(
                original_averageDensity,
                tryParse<float>(getValue(line, "[original_averageDensity : ")).value_or(0.0f));
            if (checkResults.empty()){ checkResults = getValue(line, "[" + nameToken_ + "_checkResults : "); }
        }

        if (gflops > gflops_){
            K_ = K;
            gflops_ = gflops;
            alpha_ = alpha;
            delta_ = delta;
            numClusters_ = numClusters;
            numDenseBlock_ = numDenseBlock;
            reordering_ = reordering;
            rowReordering_ = rowReordering;
            colReordering_ = colReordering;
            checkResults_ = checkResults;
        }

        const ReorderingData reordering_data = {
            gflops, numClusters, numDenseBlock, averageDensity, original_numDenseBlock,
            original_averageDensity, rowReordering, colReordering
        };
        if (alpha_to_delta_to_reorderingData_.find(alpha) == alpha_to_delta_to_reorderingData_.end()){
            std::map<float, ReorderingData> delta_to_tuple;
            delta_to_tuple[delta] = reordering_data;
            alpha_to_delta_to_reorderingData_[alpha] = delta_to_tuple;
        }
        else{
            auto& delta_to_reorderingData = alpha_to_delta_to_reorderingData_[alpha];
            if (delta_to_reorderingData.find(delta) == delta_to_reorderingData.end()){
                delta_to_reorderingData[delta] = reordering_data;
            }
            else{
                // if the reorderingData is already exist, then update the reorderingData
                if (delta_to_reorderingData[delta].gflops < gflops){
                    delta_to_reorderingData[delta] = reordering_data;
                }
            }
        }
    }

    // float gflops() const{ return gflops_; }
    int numDenseBlock() const{ return numDenseBlock_; }
    const std::string& checkResults() const{ return checkResults_; }
    float reordering() const{ return reordering_; }
    float rowReordering() const{ return rowReordering_; }
    float colReordering() const{ return colReordering_; }
    float alpha() const{ return alpha_; }
    float delta() const{ return delta_; }

    const std::map<float, std::map<float, ReorderingData>>& alphaToDeltaToReorderingData() const{
        return alpha_to_delta_to_reorderingData_;
    }

protected:
    std::string nameToken_;

    int numClusters_ = 0;
    int numDenseBlock_ = 0;
    float reordering_ = std::numeric_limits<float>::max();
    float rowReordering_ = std::numeric_limits<float>::max();
    float colReordering_ = std::numeric_limits<float>::max();
    float alpha_ = 0.0f;
    float delta_ = 0.0f;
    std::string checkResults_;

    // alpha -> delta -> (gflops, numDenseBlock, averageDensity, original_numDenseBlock, original_averageDensity, rowReorderingTime, colReorderingTime)
    std::map<float, std::map<float, ReorderingData>> alpha_to_delta_to_reorderingData_;
};

struct OneTimeData{
    void update(const std::vector<std::string>& oneTimeResults);

    BaseLineWithParameter BSMR_{"bsmr"};
    BaseLine cuSDDMM_{"cuSDDMM"};
    BaseLine cuSPARSE_{"cuSPARSE"};
    BaseLine ASpT_{"ASpT"};
    BaseLine RoDe_{"RoDe"};
    BaseLineWithParameter BSA_{"BSA"};
    BaseLine FlashSparse_{"FlashSparse"};
    BaseLine TCGNN_{"TCGNN"};
    BaseLine Sputnik_{"Sputnik"};
};

void OneTimeData::update(const std::vector<std::string>& oneTimeResults){
    BSMR_.parse(oneTimeResults);
    cuSDDMM_.parse(oneTimeResults);
    cuSPARSE_.parse(oneTimeResults);
    ASpT_.parse(oneTimeResults);
    RoDe_.parse(oneTimeResults);
    BSA_.parse(oneTimeResults);
    FlashSparse_.parse(oneTimeResults);
    TCGNN_.parse(oneTimeResults);
    Sputnik_.parse(oneTimeResults);
}

struct ResultsInformation{
    bool initInformation(const std::vector<std::string>& oneTimeResults);

    std::string file_;
    std::string M_;
    std::string N_;
    std::string NNZ_;
    std::string sparsity_;

    OneTimeData oneTimeData_;
};

bool ResultsInformation::initInformation(const std::vector<std::string>& oneTimeResults){
    std::string file, M, N, NNZ, sparsity;
    for (const std::string& line : oneTimeResults){
        file = file.empty() ? getValue(line, "[File : ") : file;
        M = M.empty() ? getValue(line, "[M : ") : M;
        N = N.empty() ? getValue(line, "[N : ") : N;
        NNZ = NNZ.empty() ? getValue(line, "[NNZ : ") : NNZ;
        sparsity = sparsity.empty() ? getValue(line, "[sparsity : ") : sparsity;
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

    oneTimeData_.update(oneTimeResults);

    return true;
}

std::string getFileName(const std::string& path){
    if (path.empty()) return "";

    const size_t pos = path.find_last_of("/\\");
    // if (pos == std::string::npos){
    //     std::cerr << "Warning. The input path has no parent folder" << std::endl;
    // }
    const std::string filename = (pos == std::string::npos) ? path : path.substr(pos + 1);
    return filename;
}

std::string getParentFolderPath(const std::string& path){
    if (path.empty()) return "";

    const size_t pos = path.find_last_of("/\\");
    // if (pos == std::string::npos){
    //     std::cerr << "Warning. The input path has no parent folder" << std::endl;
    // }
    const std::string directory = (pos == std::string::npos) ? "" : path.substr(0, pos + 1);
    return directory;
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
    const int numResults = matrixFileToResultsInformationMap.size();
    int numErrors = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        bool isCorrect = checkIsCorrect(iter.second.oneTimeData_.BSMR_.checkResults());

        if (!isCorrect){
            ++numErrors;
        }
    }
    const float accuracy = 1.0f - static_cast<float>(numErrors) / numResults;

    printf("Accuracy: %.2f%%\n", accuracy * 100);

    return accuracy;
}

auto evaluateBaseLine = [](const float BSMR_gflops,
                           const float baseLine_gflops,
                           std::vector<int>& numSpeedups,
                           float& maxSpeedup,
                           float& sumSpeedup,
                           int& numResults) -> void{
    if (BSMR_gflops <= 1e-6 || baseLine_gflops <= 1e-6){
        return;
    }
    const float speedup = BSMR_gflops / baseLine_gflops;
    maxSpeedup = std::max(speedup, maxSpeedup);
    sumSpeedup += speedup;
    ++numResults;
    // [0,1.0), [1.0,1.5), [1.5,2.0), [2.0, )
    if (speedup < 1.0f){
        ++numSpeedups[0];
    }
    if (speedup >= 1.0f && speedup < 1.5f){
        ++numSpeedups[1];
    }
    if (speedup >= 1.5f && speedup < 2.0f){
        ++numSpeedups[2];
    }
    if (speedup >= 2.0f){
        ++numSpeedups[3];
    }
};

auto evaluateHybrid = [](const float BSMR_gflops,
                         const float baseLine_gflops,
                         std::vector<int>& numSpeedups,
                         float& maxSpeedup,
                         float& sumSpeedup,
                         int& numResults) -> void{
    if (BSMR_gflops <= 1e-6 || baseLine_gflops <= 1e-6){
        return;
    }
    const float speedup = BSMR_gflops / baseLine_gflops;
    maxSpeedup = std::max(speedup, maxSpeedup);
    sumSpeedup += speedup;
    ++numResults;
    // [0.0,1.0), [1.0,1.2), [1.2,1.5), [1.5, )
    if (speedup < 1.0f){
        ++numSpeedups[0];
    }
    if (speedup >= 1.0f && speedup < 1.2f){
        ++numSpeedups[1];
    }
    if (speedup >= 1.2f && speedup < 1.5f){
        ++numSpeedups[2];
    }
    if (speedup >= 1.5f){
        ++numSpeedups[3];
    }
};

auto printEvaluateResults = [](const std::string& baseLineName,
                               const int numResults,
                               const float sumSpeedup,
                               const float maxSpeedup,
                               const std::vector<int>& numSpeedups) -> void{
    const float averageSpeedup = sumSpeedup / numResults;

    printSeparator("Evaluate BSMR With " + baseLineName + ":");
    printf("Number of results: %d\n", numResults);
    if (numResults < 1){
        printf("No results found\n");
        return;
    }
    printf("Average speedup: %.2fx, maximum speedup: %.2fx\n", averageSpeedup, maxSpeedup);

    printf("Speedup < 1.0x : %.1f%%\n", numSpeedups[0] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup 1.0~1.5x : %.1f%%\n", numSpeedups[1] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup 1.5~2.0x : %.1f%%\n", numSpeedups[2] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup >= 2.0x : %.1f%%\n", numSpeedups[3] / static_cast<float>(numResults) * 100.0f);

    const float accelerationCoverage =
        (numSpeedups[1] + numSpeedups[2] + numSpeedups[3]) / static_cast<float>(numResults) * 100.0f;
    printf("Acceleration coverage: %.1f%%\n", accelerationCoverage);

    printf("\n");
};

auto printEvaluateHybridResults = [](const std::string& baseLineName,
                                     const int numResults,
                                     const float sumSpeedup,
                                     const float maxSpeedup,
                                     const std::vector<int>& numSpeedups) -> void{
    const float averageSpeedup = sumSpeedup / numResults;

    printSeparator("Evaluate BSMR With " + baseLineName + ":");
    if (numResults < 1){
        printf("No results found\n");
        return;
    }
    printf("Number of results: %d\n", numResults);
    printf("Average speedup: %.2fx, maximum speedup: %.2fx\n", averageSpeedup, maxSpeedup);

    printf("Speedup < 1.0x : %.1f%%\n", numSpeedups[0] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup 1.0~1.2x : %.1f%%\n", numSpeedups[1] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup 1.2~1.5x : %.1f%%\n", numSpeedups[2] / static_cast<float>(numResults) * 100.0f);
    printf("Speedup >= 1.5x : %.1f%%\n", numSpeedups[3] / static_cast<float>(numResults) * 100.0f);

    const float accelerationCoverage =
        (numSpeedups[1] + numSpeedups[2] + numSpeedups[3]) / static_cast<float>(numResults) * 100.0f;
    printf("Acceleration coverage: %.1f%%\n", accelerationCoverage);

    printf("\n");
};

// return the average speedup adn the maximum speedup
void evaluateBSMRWithCuSDDMM(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // [0,1.0), [1.0,1.5), [1.5,2.0), [2.0, )
    std::vector<int> numSpeedups(4);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        const float BSMR_gflops = iter.second.oneTimeData_.BSMR_.gflops();
        const float cuSDDMM_gflops = iter.second.oneTimeData_.cuSDDMM_.gflops();

        evaluateBaseLine(BSMR_gflops, cuSDDMM_gflops, numSpeedups, maxSpeedup, sumSpeedup, numResults);
    }

    printEvaluateResults("cuSDDMM", numResults, sumSpeedup, maxSpeedup, numSpeedups);
}

// return the average speedup adn the maximum speedup
void evaluateBSMRWithCuSparse(
    std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // [0,1.0), [1.0,1.5), [1.5,2.0), [2.0, )
    std::vector<int> numSpeedups(4);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        const float BSMR_gflops = iter.second.oneTimeData_.BSMR_.gflops();
        const float cuSPARSE_gflops = iter.second.oneTimeData_.cuSPARSE_.gflops();

        evaluateBaseLine(BSMR_gflops, cuSPARSE_gflops, numSpeedups, maxSpeedup, sumSpeedup, numResults);
    }

    printEvaluateResults("cuSparse", numResults, sumSpeedup, maxSpeedup, numSpeedups);
}


// return the average speedup adn the maximum speedup
void evaluateBSMRWithASpT(
    std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // [0,1.0), [1.0,1.5), [1.5,2.0), [2.0, )
    std::vector<int> numSpeedups(4);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        const float BSMR_gflops = iter.second.oneTimeData_.BSMR_.gflops();
        const float ASpT_gflops = iter.second.oneTimeData_.ASpT_.gflops();

        evaluateBaseLine(BSMR_gflops, ASpT_gflops, numSpeedups, maxSpeedup, sumSpeedup, numResults);
    }

    printEvaluateResults("ASpT", numResults, sumSpeedup, maxSpeedup, numSpeedups);
}

// return the average speedup adn the maximum speedup
void evaluateBSMRWithRoDe(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // [0,1.0), [1.0,1.5), [1.5,2.0), [2.0, )
    std::vector<int> numSpeedups(4);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        const float BSMR_gflops = iter.second.oneTimeData_.BSMR_.gflops();
        const float RoDe_gflops = iter.second.oneTimeData_.RoDe_.gflops();

        evaluateBaseLine(BSMR_gflops, RoDe_gflops, numSpeedups, maxSpeedup, sumSpeedup, numResults);
    }

    printEvaluateResults("RoDe", numResults, sumSpeedup, maxSpeedup, numSpeedups);
}

void evaluateBSMRWithFlashSparse(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // [0,1.0), [1.0,1.5), [1.5,2.0), [2.0, )
    std::vector<int> numSpeedups(4);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        const float BSMR_gflops = iter.second.oneTimeData_.BSMR_.gflops();
        const float FlashSparse_gflops = iter.second.oneTimeData_.FlashSparse_.gflops();

        evaluateBaseLine(BSMR_gflops, FlashSparse_gflops, numSpeedups, maxSpeedup, sumSpeedup, numResults);
    }

    printEvaluateResults("FlashSparse", numResults, sumSpeedup, maxSpeedup, numSpeedups);
}

void evaluateBSMRWithTCGNN(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // [0,1.0), [1.0,1.5), [1.5,2.0), [2.0, )
    std::vector<int> numSpeedups(4);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        const float BSMR_gflops = iter.second.oneTimeData_.BSMR_.gflops();
        const float TCGNN_gflops = iter.second.oneTimeData_.TCGNN_.gflops();

        evaluateBaseLine(BSMR_gflops, TCGNN_gflops, numSpeedups, maxSpeedup, sumSpeedup, numResults);
    }

    printEvaluateResults("TCGNN", numResults, sumSpeedup, maxSpeedup, numSpeedups);
}

void evaluateBSMRWithSputnik(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    float sumSpeedup = 0.0f;
    float maxSpeedup = 0.0f;

    // [0,1.0), [1.0,1.5), [1.5,2.0), [2.0, )
    std::vector<int> numSpeedups(4);

    int numResults = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        const float BSMR_gflops = iter.second.oneTimeData_.BSMR_.gflops();
        const float Sputnik_gflops = iter.second.oneTimeData_.Sputnik_.gflops();

        evaluateBaseLine(BSMR_gflops, Sputnik_gflops, numSpeedups, maxSpeedup, sumSpeedup, numResults);
    }

    printEvaluateResults("Sputnik", numResults, sumSpeedup, maxSpeedup, numSpeedups);
}

void outputCSVFile(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap,
    const std::string& outputFilePath = "./"){
    const int k = matrixFileToResultsInformationMap.begin()->second.oneTimeData_.BSMR_.K();
    const std::string outputFileName = outputFilePath + "results_" + std::to_string(k) + ".csv";
    std::ofstream outFile(outputFileName);
    if (!outFile.is_open()){
        std::cerr << "Error, output file cannot be opened : " << outputFileName << std::endl;
        return;
    }

    outFile << "file,M,N,NNZ,Sparsity,K,BSMR,cuSDDMM,cuSPARSE,RoDe,ASpT,TCGNN,FlashSparse,Sputnik\n";

    for (const auto& iter : matrixFileToResultsInformationMap){
        const ResultsInformation& resultsInformation = iter.second;

        const OneTimeData& oneTimeData = resultsInformation.oneTimeData_;

        outFile << iter.first << ","; // Matrix File
        outFile << resultsInformation.M_ << ","; // M
        outFile << resultsInformation.N_ << ","; // N
        outFile << resultsInformation.NNZ_ << ","; // NNZ
        outFile << resultsInformation.sparsity_ << ","; // Sparsity
        outFile << k; // K
        outFile << "," << oneTimeData.BSMR_.gflops(); // BSMR
        outFile << "," << oneTimeData.cuSDDMM_.gflops(); // cuSDDMM
        outFile << "," << oneTimeData.cuSPARSE_.gflops(); // cuSPARSE
        outFile << "," << oneTimeData.RoDe_.gflops(); // RoDe
        outFile << "," << oneTimeData.ASpT_.gflops(); // ASpT
        outFile << "," << oneTimeData.TCGNN_.gflops(); // TCGNN
        outFile << "," << oneTimeData.FlashSparse_.gflops(); // FlashSparse
        outFile << "," << oneTimeData.Sputnik_.gflops(); // Sputnik
        outFile << "\n";
    }

    outFile.close();
}

void eliminateInvalidData(std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    // std::unordered_set<std::string> filePathSplit = {
    //     "./suiteSparse_dataset/3dtube/3dtube.mtx",
    //     "./suiteSparse_dataset/shermanACb/shermanACb.mtx",
    //     "./suiteSparse_dataset/ts-palko/ts-palko.mtx",
    //     "./suiteSparse_dataset/Maragal_8/Maragal_8.mtx",
    //     "./suiteSparse_dataset/Maragal_7/Maragal_7.mtx",
    //     "./suiteSparse_dataset/mult_dcop_01/mult_dcop_01.mtx",
    //     "./suiteSparse_dataset/co9/co9.mtx",
    //     "./suiteSparse_dataset/Maragal_6/Maragal_6.mtx",
    //     "./suiteSparse_dataset/gupta3/gupta3.mtx",
    //     "./suiteSparse_dataset/sx-mathoverflow/sx-mathoverflow.mtx",
    //     "./suiteSparse_dataset/sx-mathoverflow/sx-mathoverflow_C2Q.mtx", // 1
    //     "./suiteSparse_dataset/lpl3/lpl3.mtx",
    //     "./suiteSparse_dataset/route/route.mtx",
    //     "./suiteSparse_dataset/mult_dcop_02/mult_dcop_02.mtx",
    //     "./suiteSparse_dataset/mult_dcop_03/mult_dcop_03.mtx",
    //     "./suiteSparse_dataset/Reuters911/Reuters911.mtx",
    //     "./suiteSparse_dataset/EAT_RS/EAT_RS.mtx",
    //     "./suiteSparse_dataset/c-58/c-58.mtx"
    // };
    //
    // std::unordered_set<std::string> filePathSplit2 = {
    //     "./suiteSparse_dataset/jnlbrng1/jnlbrng1.mtx",
    //     "./suiteSparse_dataset/spmsrtls/spmsrtls.mtx",
    //     "./suiteSparse_dataset/fe_body/fe_body.mtx",
    //     "./suiteSparse_dataset/minsurfo/minsurfo.mtx",
    //     "./suiteSparse_dataset/delaunay_n16/delaunay_n16.mtx",
    //     "./suiteSparse_dataset/dixmaanl/dixmaanl.mtx",
    //     "./suiteSparse_dataset/gridgena/gridgena.mtx",
    //     "./suiteSparse_dataset/torsion1/torsion1.mtx",
    //     "./suiteSparse_dataset/obstclae/obstclae.mtx",
    //     "./suiteSparse_dataset/onera_dual/onera_dual.mtx",
    //     "./suiteSparse_dataset/3D_28984_Tetra/3D_28984_Tetra.mtx",
    //     "./suiteSparse_dataset/onera_dual/onera_dual.mtx"
    // };

    for (auto iter = matrixFileToResultsInformationMap.begin(); iter != matrixFileToResultsInformationMap.end();){
        const int m = tryParse<int>(iter->second.M_).value_or(0);
        const int n = tryParse<int>(iter->second.N_).value_or(0);
        const uint64_t nnz = tryParse<int>(iter->second.NNZ_).value_or(0);
        if (m <= 0 || n <= 0 || nnz <= 0){
            // printf("[bad file] : %s, M: %d, N: %d\n", iter->first.c_str(), m, n);
            iter = matrixFileToResultsInformationMap.erase(iter);
            continue;
        }

        ++iter;
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
        const int bsmr_numDenseBlock = iter.second.oneTimeData_.BSMR_.numDenseBlock();
        const int bsa_numDenseBlock = iter.second.oneTimeData_.BSA_.numDenseBlock();

        if (bsmr_numDenseBlock <= 0 && bsa_numDenseBlock <= 0){
            continue;
        }

        printf("|");
        std::cout << iter.second.NNZ_ << "|";
        std::cout << bsmr_numDenseBlock << "|";
        std::cout << bsa_numDenseBlock << "|";
        printf("\n");
    }

    printf("\n");
    printf("\n");
}

void evaluateReorderingOverhead(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap,
    const std::string& outputFilePath = "./"){
    // const std::string outputFileName = outputFilePath + "results_reordering_overhead.csv";
    // std::ofstream outFile(outputFileName);
    // if (!outFile.is_open()){
    //     std::cerr << "Error, output file cannot be opened : " << outputFileName << std::endl;
    //     return;
    // }
    // outFile << "file,alpha,M,N,NNZ,numClusters,rowReorderingTime,colReorderingTime,reorderingTime\n";

    std::map<int, int> M10K_to_numResults;
    std::map<int, int> N10K_to_numResults;

    // alpha -> m / 10000 -> (numResults, avgClusters, avgRowReorderingTime)
    std::map<float, std::map<int, std::tuple<int, int, float>>> alpha_to_M10K_to_tuple;

    std::vector<int> ratios(4, 0); // [0.0, 100.0), [100.0, 1000.0), [1000.0, 10000.0), [10000.0, )
    int numResults = 0;
    for (const auto& [file,resultsInformation] : matrixFileToResultsInformationMap){
        const int m = tryParse<int>(resultsInformation.M_).value_or(0);
        const int n = tryParse<int>(resultsInformation.N_).value_or(0);
        if (m != n){
            continue;
        }
        ++numResults;
        for (const auto& [alpha, deltaToReorderingData] : resultsInformation.oneTimeData_.BSMR_.
                                                                             alphaToDeltaToReorderingData()){
            const int numClusters = deltaToReorderingData.begin()->second.numClusters;
            const float rowReorderingTime = deltaToReorderingData.begin()->second.rowReorderingTime;
            const float colReorderingTime = deltaToReorderingData.begin()->second.colReorderingTime;

            if (alpha <= 0.0f){
                continue;
            }

            std::get<0>(alpha_to_M10K_to_tuple[alpha][m / 10000])++; // numResults
            std::get<1>(alpha_to_M10K_to_tuple[alpha][m / 10000]) += numClusters; // avgClusters
            std::get<2>(alpha_to_M10K_to_tuple[alpha][m / 10000]) += rowReorderingTime + colReorderingTime;
            // avgRowReorderingTime

            ++M10K_to_numResults[m / 10000];

            // outFile << file << ","; // file
            // outFile << alpha << ","; // alpha
            // outFile << resultsInformation.M_ << ","; // M
            // outFile << resultsInformation.N_ << ","; // N
            // outFile << resultsInformation.NNZ_ << ","; // NNZ
            // outFile << numClusters << ","; // numClusters
            // outFile << rowReorderingTime << ","; // rowReorderingTime
            // outFile << colReorderingTime << ","; // colReorderingTime
            // outFile << (rowReorderingTime + colReorderingTime) << "\n"; // reorderingTime
        }
    }
    // outFile.close();

    for (auto& [alpha, M10K_to_tuple] : alpha_to_M10K_to_tuple){
        for (auto& [M10K, tuple] : M10K_to_tuple){
            std::get<1>(tuple) /= std::get<0>(tuple); // average clusters
            std::get<2>(tuple) /= std::get<0>(tuple); // average row reordering time
        }
    }

    printSeparator("Evaluate Reordering Overhead:");
    printf("Number of results: %d\n", numResults);
    for (const auto& [alpha, M10K_to_tuple] : alpha_to_M10K_to_tuple){
        for (const auto& [M10K, tuple] : M10K_to_tuple){
            printf(
                "Alpha: %.2f, m in [%d, %d), Num Results: %d, Avg Clusters: %d, Avg Row Reordering Time: %.2f ms\n",
                alpha, M10K * 10000, (M10K + 1) * 10000,
                std::get<0>(tuple),
                std::get<1>(tuple),
                std::get<2>(tuple));
        }
    }
}

void evaluateReorderingWithBSA(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    std::map<float, std::map<float, uint32_t>> alpha_to_delta_to_BSMR_numDenseBlock;
    std::map<float, std::map<float, uint32_t>> alpha_to_delta_to_BSA_numDenseBlock;
    std::map<float, std::map<float, uint32_t>> alpha_to_delta_to_original_numDenseBlock;

    // number of times denseblock appears in all matrices
    std::map<float, std::map<float, int>> alpha_to_delta_to_numResults;

    std::map<float, std::map<float, float>> alpha_to_delta_to_BSMR_averageDensity;
    std::map<float, std::map<float, float>> alpha_to_delta_to_BSA_averageDensity;
    std::map<float, std::map<float, float>> alpha_to_delta_to_original_averageDensity;

    // number of times denseblock appears in BSMR
    std::map<float, std::map<float, int>> alpha_to_delta_to_numResultsBSMR;
    // number of times denseblock appears in BSA
    std::map<float, std::map<float, int>> alpha_to_delta_to_numResultsBSA;
    // number of times denseblock appears in original matrix
    std::map<float, std::map<float, int>> alpha_to_delta_to_numResultsOriginal;

    std::map<float, std::map<float, float>> alpha_to_delta_to_BSMR_reordering;
    std::map<float, std::map<float, float>> alpha_to_delta_to_BSA_reordering;

    for (const auto& iter : matrixFileToResultsInformationMap){
        for (const auto& pair : iter.second.oneTimeData_.BSMR_.alphaToDeltaToReorderingData()){
            const float alpha = pair.first;
            for (const auto& deltaToTuple : pair.second){
                const float delta = deltaToTuple.first;

                if (alpha == 0.0f || delta == 0.0f){
                    continue; // skip if alpha or delta is zero
                }

                const float BSMR_gflops = deltaToTuple.second.gflops;
                if (BSMR_gflops <= 1e-6){
                    continue; // skip if the result is not valid
                }

                alpha_to_delta_to_BSMR_reordering[alpha][delta] += deltaToTuple.second.rowReorderingTime +
                    deltaToTuple.second.colReorderingTime;

                const int BSMR_numDenseBlock = deltaToTuple.second.numDenseBlock;
                const float BSMR_averageDensity = deltaToTuple.second.averageDensity;
                const int original_numDenseBlock = deltaToTuple.second.original_numDenseBlock;
                const float original_averageDensity = deltaToTuple.second.original_averageDensity;

                float BSA_reordering = 0.0f;
                int BSA_numDenseBlock = 0;
                float BSA_averageDensity = 0.0f;
                try{
                    alpha_to_delta_to_BSA_reordering[alpha][delta] += iter
                                                                      .second.oneTimeData_.BSA_.
                                                                      alphaToDeltaToReorderingData()
                                                                      .at(alpha)
                                                                      .at(delta).rowReorderingTime;
                    BSA_numDenseBlock = iter.second.oneTimeData_.BSA_.alphaToDeltaToReorderingData()
                                            .at(alpha)
                                            .at(delta).numDenseBlock;
                    BSA_averageDensity = iter.second.oneTimeData_.BSA_.alphaToDeltaToReorderingData()
                                             .at(alpha)
                                             .at(delta).averageDensity;
                }
                catch (...){
                    continue;
                }

                if (BSMR_numDenseBlock <= 0 && BSA_numDenseBlock <= 0 && original_numDenseBlock <= 0){
                    continue; // skip if both BSMR and BSA have no dense blocks
                }

                alpha_to_delta_to_BSMR_numDenseBlock[alpha][delta] += BSMR_numDenseBlock;
                alpha_to_delta_to_BSA_numDenseBlock[alpha][delta] += BSA_numDenseBlock;
                alpha_to_delta_to_original_numDenseBlock[alpha][delta] += original_numDenseBlock;
                ++alpha_to_delta_to_numResults[alpha][delta];

                if (BSMR_numDenseBlock > 0){
                    alpha_to_delta_to_BSMR_averageDensity[alpha][delta] += BSMR_averageDensity;
                    ++alpha_to_delta_to_numResultsBSMR[alpha][delta];
                }

                if (BSA_numDenseBlock > 0){
                    alpha_to_delta_to_BSA_averageDensity[alpha][delta] += BSA_averageDensity;
                    ++alpha_to_delta_to_numResultsBSA[alpha][delta];
                }

                if (original_numDenseBlock > 0){
                    alpha_to_delta_to_original_averageDensity[alpha][delta] += original_averageDensity;
                    ++alpha_to_delta_to_numResultsOriginal[alpha][delta];
                }
            }
        }
    }

    printSeparator("Evaluate Reordering With BSA:");

    for (const auto& alphaToDelta : alpha_to_delta_to_BSMR_numDenseBlock){
        const float alpha = alphaToDelta.first;
        for (const auto& deltaToNumDenseBlock : alphaToDelta.second){
            const float delta = deltaToNumDenseBlock.first;

            const uint32_t BSMR_numDenseBlock = alpha_to_delta_to_BSMR_numDenseBlock[alpha][delta] /
                alpha_to_delta_to_numResults[alpha][delta];
            const uint32_t BSA_numDenseBlock = alpha_to_delta_to_BSA_numDenseBlock[alpha][delta] /
                alpha_to_delta_to_numResults[alpha][delta];
            const uint32_t original_numDenseBlock = alpha_to_delta_to_original_numDenseBlock[alpha][delta] /
                alpha_to_delta_to_numResults[alpha][delta];

            const float BSMR_averageDensity = alpha_to_delta_to_BSMR_averageDensity[alpha][delta] /
                alpha_to_delta_to_numResultsBSMR[alpha][delta];
            const float BSA_averageDensity = alpha_to_delta_to_BSA_averageDensity[alpha][delta] /
                alpha_to_delta_to_numResultsBSA[alpha][delta];
            const float original_averageDensity = alpha_to_delta_to_original_averageDensity[alpha][delta] /
                alpha_to_delta_to_numResultsOriginal[alpha][delta];

            printf("Alpha: %.2f, Delta: %.2f, "
                   "BSMR average num dense blocks: %d, "
                   "BSA average num dense blocks: %d, "
                   "Original average num dense blocks: %d, "
                   "BSMR average density: %.2f, "
                   "BSA average density: %.2f, "
                   "Original average density: %.2f, "
                   "num results: %u\n",
                   alpha, delta, BSMR_numDenseBlock, BSA_numDenseBlock, original_numDenseBlock,
                   BSMR_averageDensity, BSA_averageDensity, original_averageDensity,
                   alpha_to_delta_to_numResults[alpha][delta]);
        }
    }

    printf("\n");
}

void evaluateHybridSddmm(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap,
    const std::string& outputFilePath = "./"){
    const int k = matrixFileToResultsInformationMap.begin()->second.oneTimeData_.BSMR_.K();
    const std::string outputFileName = outputFilePath + "results_hybrid_" + std::to_string(k) + ".csv";
    std::ofstream outFile(outputFileName);
    if (!outFile.is_open()){
        std::cerr << "Error, output file cannot be opened : " << outputFileName << std::endl;
        return;
    }

    outFile << "file,M,N,NNZ,Sparsity,alpha,K,BSMR,BSMR_Only_Tensor_core,BSMR_Only_CUDA_Core\n";

    float sumSpeedupWithOnlyTensorCore = 0.0f;
    float sumSpeedupWithOnlyCudaCore = 0.0f;
    int numResults = 0;
    float maxSpeedupWithOnlyTensorCore = 0;
    float maxSpeedupWithOnlyCudaCore = 0;
    std::vector<int> numSpeedupsWithOnlyTC(4); // [0,1.0), [1.0,1.2), [1.2,1.5), [1.5, )
    std::vector<int> numSpeedupsWithOnlyCUDA(4); // [0,1.0), [1.0,1.2), [1.2,1.5), [1.5, )
    for (const auto& iter : matrixFileToResultsInformationMap){
        for (const auto& pair : iter.second.oneTimeData_.BSMR_.alphaToDeltaToReorderingData()){
            const float curAlpha = pair.first;
            const float BSMR_gflops = iter.second.oneTimeData_.BSMR_.gflops();
            float BSMR_gflops_only_Tensor_core = 0.0f;
            float BSMR_gflops_only_CUDA_core = 0.0f;
            for (const auto& deltaToTuple : pair.second){
                const float curDelta = deltaToTuple.first;
                // Only Tensor core
                if (curAlpha == iter.second.oneTimeData_.BSMR_.alpha() && curDelta == 0.0f){
                    BSMR_gflops_only_Tensor_core = deltaToTuple.second.gflops;
                }
                // Only CUDA core
                if (curAlpha == iter.second.oneTimeData_.BSMR_.alpha() && curDelta > 1.0f){
                    BSMR_gflops_only_CUDA_core = deltaToTuple.second.gflops;
                }
            }
            if (BSMR_gflops <= 1e-6 || BSMR_gflops_only_Tensor_core <= 1e-6 ||
                BSMR_gflops_only_CUDA_core <= 1e-6){
                continue; // skip if the result is not valid
            }
            evaluateHybrid(BSMR_gflops, BSMR_gflops_only_Tensor_core,
                           numSpeedupsWithOnlyTC, maxSpeedupWithOnlyTensorCore,
                           sumSpeedupWithOnlyTensorCore, numResults);
            evaluateHybrid(BSMR_gflops, BSMR_gflops_only_CUDA_core,
                           numSpeedupsWithOnlyCUDA, maxSpeedupWithOnlyCudaCore,
                           sumSpeedupWithOnlyCudaCore, numResults);
            --numResults; // Adjust for the double counting of results

            const int M = tryParse<int>(iter.second.M_).value_or(0);
            const int N = tryParse<int>(iter.second.N_).value_or(0);
            const int NNZ = tryParse<int>(iter.second.NNZ_).value_or(0);
            const float sparsity = tryParse<float>(iter.second.sparsity_).value_or(0.0f);

            outFile << iter.first << ","; // Matrix File
            outFile << M << ","; // M
            outFile << N << ","; // N
            outFile << NNZ << ","; // NNZ
            outFile << sparsity << ","; // Sparsity
            outFile << curAlpha << ","; // Alpha
            outFile << k << ","; // K
            outFile << BSMR_gflops << ","; // BSMR
            outFile << BSMR_gflops_only_Tensor_core << ","; // BSMR Only Tensor core
            outFile << BSMR_gflops_only_CUDA_core << "\n"; // BSMR Only CUDA core
        }
    }
    outFile.close();

    printEvaluateHybridResults("Only CUDA core", numResults, sumSpeedupWithOnlyCudaCore,
                               maxSpeedupWithOnlyCudaCore, numSpeedupsWithOnlyCUDA);
    printEvaluateHybridResults("Only Tensor core", numResults, sumSpeedupWithOnlyTensorCore,
                               maxSpeedupWithOnlyTensorCore, numSpeedupsWithOnlyTC);

    int numHybridBest = 0;
    for (const auto& iter : matrixFileToResultsInformationMap){
        if (iter.second.oneTimeData_.BSMR_.numDenseBlock() > 0){
            ++numHybridBest;
        }
    }
    printf("Number of hybrid best results: %d\n", numHybridBest);
}

void analyzeDataset(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    printSeparator("Dataset Analysis:");

    printf("Number of matrix files: %d\n", static_cast<int>(matrixFileToResultsInformationMap.size()));

    int maxM = 0;
    int minM = std::numeric_limits<int>::max();

    int maxN = 0;
    int minN = std::numeric_limits<int>::max();

    uint64_t maxNNZ = 0;
    uint64_t minNNZ = std::numeric_limits<int>::max();

    float minSparsity = 100.0f;
    float maxSparsity = 0.0f;

    std::map<int, int> numFilesPerM10K;
    std::map<int, int> numFilesPerN10K;

    for (const auto& iter : matrixFileToResultsInformationMap){
        const int M = tryParse<int>(iter.second.M_).value_or(0);
        const int N = tryParse<int>(iter.second.N_).value_or(0);
        const uint64_t NNZ = tryParse<int>(iter.second.NNZ_).value_or(0);
        const float sparsity = tryParse<float>(iter.second.sparsity_).value_or(0.0f);

        maxM = std::max(M, maxM);
        minM = std::min(M, minM);

        maxN = std::max(N, maxN);
        minN = std::min(N, minN);

        maxNNZ = std::max(NNZ, maxNNZ);
        minNNZ = std::min(NNZ, minNNZ);

        minSparsity = std::min(sparsity, minSparsity);
        maxSparsity = std::max(sparsity, maxSparsity);

        numFilesPerM10K[M / 10000]++;
        numFilesPerN10K[N / 10000]++;
    }

    printf("Minimum m: %d, maximum m: %d\n", minM, maxM);
    printf("Minimum n: %d, maximum n: %d\n", minN, maxN);
    printf("Minimum nnz: %lu, maximum nnz: %lu\n", minNNZ, maxNNZ);
    printf("Minimum sparsity: %.2f%%, maximum sparsity: %.2f%%\n", minSparsity, maxSparsity);
    for (const auto& iter : numFilesPerM10K){
        printf("Number of files with m in [%d, %d): %d\n", iter.first * 10000, (iter.first + 1) * 10000,
               iter.second);
    }
    for (const auto& iter : numFilesPerN10K){
        printf("Number of files with n in [%d, %d): %d\n", iter.first * 10000, (iter.first + 1) * 10000,
               iter.second);
    }

    printf("\n");
}

void analyzeParameters(
    const std::unordered_map<std::string, ResultsInformation>& matrixFileToResultsInformationMap){
    printSeparator("BSMR Parameter Analysis:");


    std::unordered_map<float, int> alphaToNumResults;
    std::unordered_map<float, int> deltaToNumResults;

    std::unordered_map<float, std::unordered_map<float, int>> alphaToDeltaToNumResults;

    for (const auto& iter : matrixFileToResultsInformationMap){
        const float alpha = iter.second.oneTimeData_.BSMR_.alpha();
        const float delta = iter.second.oneTimeData_.BSMR_.delta();
        if (alphaToNumResults.find(alpha) == alphaToNumResults.end()){
            alphaToNumResults[alpha] = 0;
        }
        if (deltaToNumResults.find(delta) == deltaToNumResults.end()){
            deltaToNumResults[delta] = 0;
        }
        ++alphaToNumResults[alpha];
        ++deltaToNumResults[delta];
        ++alphaToDeltaToNumResults[alpha][delta];
    }

    float modeAlpha = 0.0f;
    int maxNumAlpha = 0;

    float modeDelta = 0.0f;
    int maxNumDelta = 0;

    std::array<float, 2> modeAlphaDelta;
    int maxNumAlphaDelta = 0;
    for (const auto& iter : alphaToNumResults){
        if (iter.second > maxNumAlpha){
            maxNumAlpha = iter.second;
            modeAlpha = iter.first;
        }
    }
    for (const auto& iter : deltaToNumResults){
        if (iter.second > maxNumDelta){
            maxNumDelta = iter.second;
            modeDelta = iter.first;
        }
    }
    for (const auto& [alpha, deltaToNum] : alphaToDeltaToNumResults){
        for (const auto& [delta, Num] : deltaToNum){
            if (Num > maxNumAlphaDelta){
                maxNumAlphaDelta = Num;
                modeAlphaDelta[0] = alpha;
                modeAlphaDelta[1] = delta;
            }
        }
    }
    printf("Mode alpha: %.2f, number of results: %d\n", modeAlpha, maxNumAlpha);
    printf("Mode delta: %.2f, number of results: %d\n", modeDelta, maxNumDelta);

    printf("Mode alpha and delta: (%.2f, %.2f), number of results: %d\n", modeAlphaDelta[0], modeAlphaDelta[1],
           maxNumAlphaDelta);

    printf("\n");
}

int main(const int argc, const char* argv[]){
    // Read the results file
    std::string resultsPath = "results/";
    SettingInformation settingInformation;
    std::unordered_map<std::string, ResultsInformation> matrixFileToResultsInformationMap;
    for (int fileIdx = 1; fileIdx < argc; ++fileIdx){
        resultsPath = getParentFolderPath(argv[fileIdx]);
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
            const std::string fileName = getFileName(matrixFile);
            const std::string key = matrixFile;
            if (matrixFileToResultsInformationMap.find(key) == matrixFileToResultsInformationMap.end()){
                matrixFileToResultsInformationMap[key] = ResultsInformation();
            }
            if (!matrixFileToResultsInformationMap[key].initInformation(oneTimeResults)){
                return -1;
            }
        }
    }
    eliminateInvalidData(matrixFileToResultsInformationMap);

    analyzeDataset(matrixFileToResultsInformationMap);

    calculateAccuracy(matrixFileToResultsInformationMap);

    analyzeParameters(matrixFileToResultsInformationMap);

    evaluateBSMRWithCuSparse(matrixFileToResultsInformationMap);

    evaluateBSMRWithCuSDDMM(matrixFileToResultsInformationMap);

    evaluateBSMRWithASpT(matrixFileToResultsInformationMap);

    evaluateBSMRWithTCGNN(matrixFileToResultsInformationMap);

    evaluateBSMRWithSputnik(matrixFileToResultsInformationMap);

    evaluateBSMRWithRoDe(matrixFileToResultsInformationMap);

    evaluateBSMRWithFlashSparse(matrixFileToResultsInformationMap);

    outputCSVFile(matrixFileToResultsInformationMap, resultsPath);

    evaluateHybridSddmm(matrixFileToResultsInformationMap, resultsPath);

    evaluateReorderingWithBSA(matrixFileToResultsInformationMap);

    evaluateReorderingOverhead(matrixFileToResultsInformationMap, resultsPath);

    return 0;
}
