# sddmm-gpu

---

Sampled Dense-Dense Matrix Multiplication(SDDMM) on GPU

$\mathbf{C}=\alpha(op(\mathbf{A})\cdot op(\mathbf{B}))\circ spy(\mathbf{C})+\beta\mathbf{C}$

---

## Input format

Supoprts Matrix Market (https://sparse.tamu.edu/about) input format. Example avaialable in dataset folder.
Conversion code from col sorted to row sorted can be found inside sddmm.cu

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

- `-k` : K value. k is a multiple of 32
- `-f` : input file path
- `-a` : alpha value
- `-b` : beta value

Example :

```shell
./sddmm-gpu -k 256 -f ../dataset/nips.mtx -a 1.0 -b 0.0
```