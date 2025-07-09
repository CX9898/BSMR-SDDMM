#!/bin/bash

git clone https://github.com/CX9898/dlmc-dataset.git

bash make_matrices_list.sh dlmc-dataset

python convert_smtx_to_mtx.py dlmc-dataset/matrix_file_list_smtx.txt

bash make_matrices_list.sh dlmc-dataset
