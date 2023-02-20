# data-adaptive-controller-learning

To reproduce the plots in the ICML-submitted version:

1. Create the Anaconda environment:

    conda env create -f environment.yml
    conda activate gaps

    or, get the required packages some other way.

2. Run the experiments in parallel. NOTE: Will take a while.

    make -j

3. Examine output in the `Plots/` directory.
