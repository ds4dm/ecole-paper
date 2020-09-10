# Ecole vs Gasse et al. 2019

This repository contains the code to compare Ecole and
*[Exact combinatorial optimization with graph convolutional neural networks](http://papers.nips.cc/paper/9690-exact-combinatorial-optimization-with-graph-convolutional-neural-networks)*
Gasse, Chételat, Ferroni, Charlin, and Lodi (2019) in Advances in Neural Information Processing Systems (pp. 15580-15592).

## Setup
```bash
git submodule update --init --recursive
conda create --name ecole_vs_gasse --file environment.yaml
conda env update --file vendor/ecole/conda-dev.yml
cmake -B ecole_build -S vendor/ecole -D CMAKE_BUILD_TYPE=Release
pip install ecole_build/python
pip install vendor/PySCIPOpt-Gasse/
pip install .
```
