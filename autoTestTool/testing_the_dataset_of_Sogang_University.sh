#!/bin/bash

sh download_dlmc_dataset.sh

bash build_program.sh

bash testTool.sh -f ../dataset/dlmc/rn50_matrices.txt -p ./build_zcx/sddmm-gpu -n zcx_results
bash testTool.sh -f ../dataset/dlmc/transformer_matrices.txt -p ./build_zcx/sddmm-gpu -n zcx_results_1

bash testTool.sh -f ../dataset/dlmc/rn50_matrices.txt -p ./build_isratnisa/isratnisa-sddmm -n isratnisa_results
bash testTool.sh -f ../dataset/dlmc/transformer_matrices.txt -p ./build_isratnisa/isratnisa-sddmm -n isratnisa_results_1
