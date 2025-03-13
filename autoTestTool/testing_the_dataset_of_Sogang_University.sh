#!/bin/bash

sh download_dlmc_dataset.sh

bash build_program.sh

bash testTool.sh -f ../dataset/dlmc/rn50_matrices.txt -p ${zcx_program_path}${zcx_program_name} -n zcx_results
bash testTool.sh -f ../dataset/dlmc/transformer_matrices.txt -p ${zcx_program_path}${zcx_program_name} -n zcx_results_1

bash testTool.sh -f ../dataset/dlmc/rn50_matrices.txt -p ${isratnisa_program_path}${isratnisa_program_name} -n isratnisa_results
bash testTool.sh -f ../dataset/dlmc/transformer_matrices.txt -p ${isratnisa_program_path}${isratnisa_program_name} -n isratnisa_results_1
