all: Plots/dis_vs_cts_regret.pdf Plots/pendulum_params.pdf Plots/pendulum_costs.pdf Plots/lambda_confident_comparison.pdf

Plots/pendulum_params.pdf: pendulum_plot_params.py Data/pendulum_gaussian.npz Data/pendulum_walk.npz
	mkdir -p Plots
	python $^

Plots/pendulum_costs.pdf: pendulum_plot_costs.py Data/pendulum_gaussian.npz Data/pendulum_walk.npz
	mkdir -p Plots
	python $^

Plots/dis_vs_cts_regret.pdf: discrete_vs_cts_plot.py Data/discrete_vs_cts.npz
	mkdir -p Plots
	python $<

Plots/lambda_confident_comparison.pdf: lambda_confident_comparison.py
	mkdir -p Plots
	python $<

Data/pendulum_gaussian.npz: pendulum.py
	mkdir -p Data
	python $< $@

Data/pendulum_walk.npz: pendulum.py
	mkdir -p Data
	python $< --walk $@

Data/discrete_vs_cts.npz: discrete_vs_cts.py
	mkdir -p Data
	python $<

clean:
	rm -f Plots/*.pdf
	rm -f Data/*.npz
