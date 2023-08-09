# Li-ion Battery Prognosis Based on Hybrid Bayesian PINN
Code for the paper:

Nascimento, Viana, Corbetta, Kulkarni. A Framework for Li-ion Battery Prognosis Based on Hybrid Bayesian Physics-Informed Neural Networks
Nature Scientific Reports, 2023.

## Summary
Li-ion batteries are the main power source used in electric propulsion applications (e.g., electric cars, unmanned aerial vehicles, and advanced air mobility aircraft).
Analytics-based monitoring and forecasting for metrics such as state of charge and state of health based on battery-specific usage data are critical to ensure high reliability levels.
However, the complex electrochemistry that governs battery operation leads to computationally expensive physics-based models; which become unsuitable for prognosis and health management applications.
We propose a hybrid physics-informed machine learning approach that simulates dynamical responses by directly implementing numerical integration of principle-based governing equations through recurrent neural networks.
While reduced-order models describe part of the voltage discharge under constant or variable loading conditions, model-form uncertainty is captured through multi-layer perceptrons and battery-to-battery aleatory uncertainty is modeled through variational multi-layer perceptrons.
In addition, we use a Bayesian approach to merge fleet-wide data in the form of priors with battery-specific discharge cycles, where the battery capacity is fully available or only partially available.
We illustrate the effectiveness of our proposed framework using the NASA Prognostics Data Repository Battery dataset, which contains experimental discharge data on Li-ion batteries obtained in a controlled environment.

## Credits
This work is the result of Dr. Renato Nascimento's work, carried out as a collaboration between Prof. Felipe Viana's Probabilistic Mechanics Laboratory at the University of Central Florida (UCF), and KBR Inc., at NASA Ames Research Center, Calif., where Renato was an intern during 2020.

[Probabilistic Mechanics Laboratory (PML)](https://github.com/PML-UCF)









