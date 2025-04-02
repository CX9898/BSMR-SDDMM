#!/bin/bash

wget -nc https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz

tar -xvf dlmc.tar.gz

results_path="dataset_of_Sogang_University_results/"
rn50_results_path="${results_path}rn50_results/"
transformer_results_path="${results_path}transformer_results/"

mkdir -p ${results_path}
mkdir -p ${rn50_results_path}
mkdir -p ${transformer_results_path}

mv -n dlmc ${results_path}
mv -n ${results_path}2048_512_dlmc_data.txt dataset/dlmc/

bash build_program.sh

bash testTool.sh -f ${results_path}dlmc/rn50_matrices.txt -p ./build_zcx/sddmm-gpu -n ${rn50_results_path}zcx_results_rn50
bash testTool.sh -f ${results_path}dlmc/transformer_matrices.txt -p ./build_zcx/sddmm-gpu -n ${transformer_results_path}zcx_results_transformer

bash testTool.sh -f ${results_path}dlmc/rn50_matrices.txt -p ./build_isratnisa/isratnisa-sddmm -n ${rn50_results_path}isratnisa_results_rn50
bash testTool.sh -f ${results_path}dlmc/transformer_matrices.txt -p ./build_isratnisa/isratnisa-sddmm -n ${transformer_results_path}isratnisa_results_transformer

# 分析结果
g++ autoAnalysisResults.cpp -o autoAnalysisResults
./autoAnalysisResults "${rn50_results_path}zcx_results_rn50.log" "${rn50_results_path}isratnisa_results_rn50.log" > ${rn50_results_path}analysisResults_rn50.log
echo "Results analysis completed: ${rn50_results_path}analysisResults_rn50.log"

./autoAnalysisResults "${transformer_results_path}zcx_results_transformer.log" "${transformer_results_path}isratnisa_results_transformer.log" > ${transformer_results_path}analysisResults_transformer.log
echo "Results analysis completed: ${transformer_results_path}analysisResults_transformer.log"

# 结果可视化
python3 resultsVisualization.py -file ${rn50_results_path}analysisResults_rn50.log -outdir ${rn50_results_path}
python3 resultsVisualization.py -file ${transformer_results_path}analysisResults_transformer.log -outdir ${transformer_results_path}
