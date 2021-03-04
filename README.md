# Ecole Paper
This repository contains the conda for the Ecole paper, including comparison with
*[Exact combinatorial optimization with graph convolutional neural networks](http://papers.nips.cc/paper/9690-exact-combinatorial-optimization-with-graph-convolutional-neural-networks)*
Gasse, Chételat, Ferroni, Charlin, and Lodi (2019) in Advances in Neural Information Processing Systems (pp. 15580-15592).

## Setup
To install Ecole and all the dependencies, run
```bash
git submodule update --init --recursive
conda env create --name ecole-paper --file environment.yaml
conda activate ecole-paper
conda env update --file "vendor/ecole/dev/conda.yml"
cmake -B ecole_build -S vendor/ecole -D ECOLE_BUILD_BENCHMARKS=ON -D CMAKE_BUILD_TYPE=Release
cmake --build ecole_build --parallel
pip install ecole_build/python
pip install .
```

## Generating the data
For benchmarking the Ecole overhead:
```bash
./build/libecole/benchmarks/benchmark-libecole --ipg 375 --nl 100 --seed 42 >> data/benchmark-branching.csv
```

For benchmarking the observation functions:
```bash
python -u -m ecole_vs_gasse.bench_observation --nl 100 --ipg 35 --seed 740 >> data/benchmark-observation.csv
```

## Visualizing the results
The notebook [`Analysis.ipynb`](Analysis.ipynb) provide code to analyse the results and reproduce the table of the
paper.
