# GMOEA

The official implementation of GMOEA [1] with Pytorch:

GMOEA.py is the main file.

GAN_model.py is the GAN model used in GMOEA.

global_parameter.py is the class including the general parameters for running GMOEA.

mating_selection.py is the mating selection strategy used in GMOEA. Specifically, the mating selection strategy in RSEA [2] is used.

spea2_env.py is the environmental selection strategy used in GMOEA. Specifically, the environmental selection strategy in SPEA2 [3] is used.

NDSort.py is the non-dominated sorting method using the efficient non-dominated sorting method in [4].

EAreal.py is the simulated binary crossover and polynomial mutation.

PM_mutation.py is the polynomial mutation [5].

tournament.py is the K-tournament selection [6].

[1] He C, Huang S, Cheng R, Tan KC, Jin Y. Evolutionary Multiobjective Optimization Driven by Generative Adversarial Networks (GANs). IEEE Transactions on Cybernetics. 2020 Apr 30.

[2]. He C, Tian Y, Jin Y, et al. A radial space division based evolutionary algorithm for many-objective optimization[J]. Applied Soft Computing, 2017, 61: 603-621.

[3]. E. Ziztler, M. Laumanns, and L. Thiele, “SPEA2: Improving the strength Pareto evolutionary algorithm for multiobjective optimization,” Evolutionary Methods for Design, Optimization, and Control, pp. 95–100, 2002.

[4]. Zhang X, Tian Y, Cheng R, et al. An efficient approach to nondominated sorting for evolutionary multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(2): 201-213.

[5]. Deb K, Beyer H G. Self-adaptive genetic algorithms with simulated binary crossover. Secretary of the SFB 531, 1999.

[6]. Xie H, Zhang M. Tuning Selection Pressure in Tournament Selection[M]. School of Engineering and Computer Science, Victoria University of Wellington, 2009.
