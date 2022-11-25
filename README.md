# ACS-1-Armageddon

Armageddon is a Python package that predicts the fate of small asteroids entering Earth’s atmosphere. The Armageddon package can be used to demonstrate a hazard map for an impact over the UK for known asteroids. It is designed to be appropriate for use in emergency response and evacuation planning.

Each software module is provided with full source code, example of usage, and automated tests.

## Overview

The underlying mathematics employs the dynamics of an asteroid in Earth’s atmosphere prior to break-up as a system of ordinary differential equations. It takes into consideration the characteristics of a given asteroid such as its initial mass, speed, trajectory angle, and internal strength. The solutions are then used to predict the airblast damage on the ground. The package returns the postcodes and population affected in England and Wales.

For further information on the project specfication, see refer to the notebooks: [ProjectDescription.ipynb](https://github.com/ese-msc-2022/acs-armageddon-Dimorphos/blob/main/ProjectDescription.ipynb), [AirburstSolver.ipynb](https://github.com/ese-msc-2022/acs-armageddon-Dimorphos/blob/main/AirburstSolver.ipynb) and [DamageMapper.ipynb](https://github.com/ese-msc-2022/acs-armageddon-Dimorphos/blob/main/DamageMapper.ipynb).

## Documentation and Usage

See [pdf documentation](https://github.com/ese-msc-2022/acs-armageddon-Dimorphos/blob/main/docs/armageddon.pdf).

See the [user_manual.ipynb](https://github.com/ese-msc-2022/acs-armageddon-Dimorphos/blob/main/examples/user_manual.ipynb) notebook file for usage demonstrations.

## Installation

To install the module and any pre-requisites, from the base directory run
```
pip install -r requirements.txt
pip install -e .
```  

To download the postcode data for England and Wales, run
```
python download_data.py
```

## Contents

### Repository Architecture

* **armageddon/** All the main functions
* **docs/** Package documentation
* **examples/** Earth example and interface notebooks for users
* **images/** Relevant images for supplementary information
* **resources/** Tables for examples and postcode and population
* **tests/** Automated testing

## How the package works

The **armageddon** folder contains all the main functions for problem computation. The two major calculation functions are `solver.py` and `damage.py`, where the former solves the system of ordinary differential equations for a given asteroid and the latter predicts the hazardous impact of such an asteroid. 

The solver is capable of implementing two different methods for solving an ODE system, the 4th-Order Runge Kutta and the Forward-Euler algorithms. Then, by employing the trajectory of the asteroid, the tool can predict the airburst events and the airburst energy.

Within the **armageddon** folder, `extension.py` demonstrates the following extended capabilities of the solver tool:

* Use a tabulated atmospheric density profile instead of an exponential atmosphere
* Determine asteroid parameters that best fit an observed energy deposition curve

The results from the solver tool can be used for hazard analysis with the damage mapper tool. The main cause of damage close to the impact site is a strong pressure in the air, or equivalently, an airblast. The damage mapper tool takes this pressure as a function of explosion energy to measure the degrees of damage.

The package's damage mapper tool also exhibits the following extended capabilities, which are included in `mapping.py` and `damage.py` respectively:

* Present the software output on a map to indicate the area of damage as a circle
* Perform a simple uncertainty analysis 

## Example usage

To get started, the user is recommended to use the example dataset which comes with the package. 

Within the example folder, see `example.py`:
```
python examples/example.py
```

A demonstration of the asteroid impact risk tool with a scenario of a large asteroid colliding with Earth over the midlands of the UK.
See [BREAKING_NEWS.ipynb](https://github.com/ese-msc-2022/acs-armageddon-Dimorphos/blob/main/examples/BREAKING_NEWS.ipynb)

## Automated testing

To run the pytest test suite, from the base directory run
```
pytest tests/
```

## GUI tool

Use command below to utilize GUI to help you visualize how a specific asteroids would influence the earth.

```
python gui/interface.py
```

## Copyright and License

Licensed under the MIT license.

See [License](https://github.com/ese-msc-2022/acs-armageddon-Dimorphos/blob/main/LICENSE.md).
