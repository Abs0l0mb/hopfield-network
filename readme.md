# Associative Memory with Hebbian Learning

## Overview

This Python project implements an associative memory model using Hebbian learning and demonstrates its capabilities through synchronous and asynchronous dynamics. The program utilizes Hopfield networks to store and recall patterns, showcasing the associative memory property.

## Features

- **Pattern Generation:** Create a set of patterns, including a checkerboard pattern, using Hebbian learning principles.
- **Perturbation:** Simulate distorted or noisy inputs by perturbing one of the patterns.
- **Synchronous Dynamics:** Apply Hebbian learning synchronously to synchronize the perturbed pattern with memorized patterns.
- **Asynchronous Dynamics:** Apply Hebbian learning asynchronously to demonstrate asynchronous synchronization.
- **Visualization:** Save the dynamic evolution of states as GIFs for both synchronous and asynchronous dynamics.
- **Energy:** Energy is a form of network performance metric. The more the network converges toward a memorized pattern, the lower its energy. We began to program certains functions that could help with it, but this work is not achieved.

## Usage

1. Run the CheckerBoard script to generate the pattern, perturb it, and perform synchronous and asynchronous dynamics.
2. Check the saved GIFs (`hebbianSync.gif` and `hebbianAsync.gif`) to visualize the associative memory dynamics.

## Requirements

- Python 3.x
- NumPy
- Matplotlib