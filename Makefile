all: pendulum_params.pdf pendulum_costs.pdf

DATA: pendulum_gaussian.npz pendulum_walk.npz

pendulum_params.pdf: pendulum_plot_params.py DATA
	python $^

pendulum_costs.pdf: pendulum_plot_costs.py DATA
	python $^

pendulum_gaussian.npz: pendulum.py
	python pendulum.py $@

pendulum_walk.npz: pendulum.py
	python pendulum.py --walk $@
