# Nonclassicality in Quantum Communication Networks

*A numerical analysis of the behaviors of quantum and classical communication networks.*

This codebase supplements our ["Nonclassicality in Quantum Communication Networks"](broken_arxiv_url) paper
with numerical results and tools for reproducing our framework. To promote transparency and reproducibility of our work, all scripts are designed to be run on a laptop computer within a couple of minutes. Future work can extend our computations to larger systems in a variety of ways as discuss in our main paper.


## Quick Start

Run a script called `multi-access-channel-vertices.jl`:
1. Navigate to the project directory `/multi-access-channels/`
2. Run `$ julia --project=. script/multi-access-channel-vertices.jl`


## Project Structure

This project combines tools in Julia and Python.
* We use Julia to derive facet inequalities that bound classical network polytopes. Our Julia computation relies on the following supporting packages: PoRTA, XPORTA.jl, Polyhedra.jl, HiGHS.jl, JuMP.jl, and BellSCenario.jl
* We use Python for to apply our variational quantum optimization techniques used to find quantum violations, as well as for plotting and data analysis. Our variational optimization software uses the following supporting packages: PennyLane, QNetVO, NumPy, Autograd, and Dask. We use Matplotlib to plot our numerical data.

Our Julia codebase is found in the `julia/` directory. Functions for enumerating the vertices of classical network polytopes are found in the `julia/src/` directory. Computations of the facet inequalities of classical network polytopes are found in the `julia/script/` directory.

Our Python codebase is found in the `pyhon/` directory. The `python/src/` directory houses our functions for applying variational optimization to maximize nonclassicality. The `python/script/` directory holds our variational optimization scripts. Each script corresponds to a fixed network in which the several different quantum resource configurations and nonclassicality witnesses are considered. Note that the scripts
assume that a few CPU cores are available for computation, this number can be adjusted by manually editing the scripts. 

The `data/` directory holds our numerical results from variational optimization. Each subdirectory of `data/` is named for a script in `python/script/`. The names of the contained files notate a quantum resource configuration and the considered nonclassicality. The `notebook/` directory contains our data analysis and visualization. 
