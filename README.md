# Nonclassicality in Quantum Communication Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10780790.svg)](https://doi.org/10.5281/zenodo.10780790)

*A numerical analysis of the behaviors of quantum and classical communication networks.*

This codebase supplements our ["Nonclassicality in Quantum Communication Networks"](broken_arxiv_url) paper
with numerical results and tools for reproducing our framework. To promote transparency and reproducibility of our work, all scripts are designed to be run on a laptop computer within a couple of minutes. Future work can extend our computations to larger systems in a variety of ways as discuss in our main paper.


## Quick Start

This project combines tools in Julia and Python.
* We use Julia to derive facet inequalities that bound classical network polytopes. Our Julia computation relies on the following supporting packages: PoRTA, XPORTA.jl, Polyhedra.jl, HiGHS.jl, JuMP.jl, and BellSCenario.jl
* We use Python for to apply our variational quantum optimization techniques used to find quantum violations, as well as for plotting and data analysis. Our variational optimization software uses the following supporting packages: PennyLane, QNetVO, NumPy, Autograd, and Dask. We use Matplotlib to plot our numerical data.
The respective dependencies are managed using the Julia package manager and Conda. To run our code, follow the instructions below to locally set up the Julia and Python enviroments.

### Julia

Julia is a programming language for high-performance computing. Follow the [Julia Installation Instructions](https://julialang.org/downloads/)
to download and install the latest version of Julia.
To install the julia dependencies, open a julia prompt and run the command:

```
$ julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

To run a Julia script, navigate to the root project directory and run the command:

```
$ julia --project=. julia/script/name_of_script.jl
```

Note that the `--project=.` flag sets up the Julia environment to use the dependencies in the `Project.toml` file.

### Python

To run a Python script or notebook, we use Conda to manage the environment:

1. Visit the [Conda Installation Page](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#installation) for details on how
   to install the `conda` command line tool.
2. To set up the development environment, navigate to the root project directory and run the command:

```
(base) $ conda env create -f environment.yml
```

3. Then activate the `network-nonclassicality-dev` environment using the command:

```
(base) $ conda activate network-nonclassicality-dev
```

4. Once the `network-nonclassicality-dev` environment is set up, a Python script can be run using the command:

```
(network-nonclassicality-dev) $ python python/script/name_of_script.py
```

## Project Structure

Our Julia codebase is found in the `julia/` directory. Functions for enumerating the vertices of classical network polytopes are found in the `julia/src/` directory. Computations of the facet inequalities of classical network polytopes are found in the `julia/script/` directory.

Our Python codebase is found in the `pyhon/` directory. The `python/src/` directory houses our functions for applying variational optimization to maximize nonclassicality. The `python/script/` directory holds our variational optimization scripts. Each script corresponds to a fixed network in which the several different quantum resource configurations and nonclassicality witnesses are considered. Note that the scripts
assume that a few CPU cores are available for computation, this number can be adjusted by manually editing the scripts. 

The `data/` directory holds our numerical results from variational optimization. Each subdirectory of `data/` is named for a script in `python/script/`. The names of the contained files notate a quantum resource configuration and the considered nonclassicality. The `notebook/` directory contains our data analysis and visualization.

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10780790.svg)](https://doi.org/10.5281/zenodo.10780790)

Please cite this software if you find it useful to your work.

```
@Misc{nonclassicality_in_communication_networks_supplemental_software,
  author        = {Brian Doolittle},
  title         = {Supplementary Code for Operational Nonclassicality in Quantum Communication Networks},
  howpublished  = {https://github.com/ChitambarLab/nonclassicality-in-quantum-communication-networks-supplemental-code},
  month         = {Feb},
  year          = {2024},
  archiveprefix = {arXiv},
  url           = {https://github.com/ChitambarLab/nonclassicality-in-quantum-communication-networks-supplemental-code},
  version       = {v0.1.0},
  doi = {https://doi.org/10.5281/zenodo.10780790}
}
```
