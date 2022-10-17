all: pendulum_states.pdf pendulum_params.pdf pendulum_costs.pdf

DATA: pendulum_linear_gaussian.npz pendulum_linear_walk.npz pendulum_nonlinear_gaussian.npz pendulum_nonlinear_walk.npz

pendulum_states.pdf: pendulum_plot_states.py DATA
	python $^

pendulum_params.pdf: pendulum_plot_params.py DATA
	python $^

pendulum_costs.pdf: pendulum_plot_costs.py DATA
	python $^

pendulum_nonlinear_gaussian.npz: pendulum.py
	python pendulum.py --nonlinear $@

pendulum_nonlinear_walk.npz: pendulum.py
	python pendulum.py --nonlinear --walk $@

pendulum_linear_gaussian.npz: pendulum.py
	python pendulum.py $@

pendulum_linear_walk.npz: pendulum.py
	python pendulum.py --walk $@
