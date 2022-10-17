all: pendulum_gaussian.pdf pendulum_walk.pdf

pendulum_%.pdf: pendulum_plot.py pendulum_%.npz
	python $^ $@

pendulum_gaussian.npz: pendulum.py
	python pendulum.py $@

pendulum_walk.npz: pendulum.py
	python pendulum.py --walk $@
