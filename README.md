# BSMR-SDDMM

---

Sampled Dense-Dense Matrix Multiplication(SDDMM) on GPU

SDDMM operates:

$\mathbf{P}_{ij} = (\mathbf{A} \cdot \mathbf{B})\_{ij} \cdot \mathbf{S}\_{ij}, \quad \text{only if} \quad \mathbf{S}\_{ij} > 0$

---

## Input format

Supports Matrix Market (https://sparse.tamu.edu/about) input format (Suffix: `.mtx`).

---

## Build requirements:

- C++ compiler with C++17 support
- CUDA SDK $\ge$ 12.0
- OpenMP
- cmake $\ge$ 3.26
- NVIDIA GPU with sm $\ge$ 80

---

## Build

Linux:

```shell
mkdir build
cd build
cmake ..
make -j
```

---

## Run

Options:

- `-f` : Input file path
- `-k` : K value. K must be a multiple of 32 (Default 32)
- `-a` : Row similarity threshold alpha (Default 0.3)
- `-d` : Block density threshold delta (Default 0.3)

Example :

```shell
./BSMR-sddmm -f ../dataset/nips.mtx -k 128
```

or

```shell
./BSMR-sddmm ../dataset/nips.mtx 128
```

## Build baselines

```shell
cd scripts/
bash build_program.sh
bash build_TCGNN.sh
bash build_FlashSparse.sh
```

## Preparing dataset

```shell
cd scripts/
bash download_suiteSparse_dataset.sh
python exclude_invalid_dataset.py
```

## Run tests

```shell
cd scripts/
bash run_all.sh
```

## Reproduce results figures

```shell
cd scripts/
bash plot_fig_5.sh
bash plot_fig_6.sh
bash plot_fig_7.sh
```
