# sddmm-gpu

---

Sampled Dense-Dense Matrix Multiplication(SDDMM) on GPU

$\mathbf{P}_{ij} = (\mathbf{A} \cdot \mathbf{B})_{ij} \cdot \mathbf{S}_{ij}, \quad \text{only if } \mathbf{S}_{ij} > 0$

---

## Input format

Supoprts Matrix Market (https://sparse.tamu.edu/about) input format.

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
cd <sddmm-gpu>
mkdir build
cd build
cmake ..
```

---

## Run

Options:

- `-f` : Input file path
- `-k` : K value. K must be a multiple of 32 (Default 32)
- `-a` : Row similarity threshold alpha (Default 0.3)
- `-b` : Column non-zero threshold beta (Default 4)

Example :

```shell
./sddmm-gpu -f ../dataset/nips.mtx -k 256
```

or

```shell
./sddmm-gpu ../dataset/nips.mtx 256
```
