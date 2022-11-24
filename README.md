# ACS-1-Armageddon

Armageddon is a Python package that predicts the fate of small asteroids entering Earth’s atmosphere. The Armageddon package can be used to demonstrate a hazard map for an impact over the UK for known asteroids. 

Each software module is provided with full source code, example of usage, and automated tests.

## How the package works

The underlying mathematics employs the dynamics of an asteroid in Earth’s atmosphere prior to break-up as a system of ordinary differential equations. It takes into consideration the characteristics of a given asteroid such as its initial mass, speed, trajectory angle, and internal strength.

The solutions are then used to predict the airblast damage on the ground and the postcodes and population affected.

## Installation

To install the module and any pre-requisites, from the base directory run
```
pip install -r requirements.txt
pip install -e .
```  

## Downloading postcode data

To download the postcode data
```
python download_data.py
```

## Automated testing

To run the pytest test suite, from the base directory run
```
pytest tests/
```

Note that you should keep the tests provided, adding new ones as you develop your code. If any of these tests fail it is likely that the scoring algorithm will not work.

## Documentation

To generate the documentation (in html format)
```
python -m sphinx docs html
```

See the `docs` directory for the preliminary documentation provided that you should add to.

## Example usage

For example usage see `example.py` in the examples folder:
```
python examples/example.py
```

## More information

For more information on the project specfication, see the python notebooks: `ProjectDescription.ipynb`, `AirburstSolver.ipynb` and `DamageMapper.ipynb`.
