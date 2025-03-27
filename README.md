# Physical Property

![PyPI version](https://badge.fury.io/py/physical-properties.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

`physical_property` is a Python package for managing physical properties with unit conversions. It provides a flexible `PhysicalProperty` base class and a variety of subclasses (e.g., `Time`, `Length`, `Temperature`, `Pressure`, `Density`, `Viscosity`, `ReynoldsNumber`) to handle dimensional quantities, unit conversions, and engineering calculations. Built with scientific computing in mind, it integrates with NumPy, SciPy, and Plotly for robust functionality.

## Features

- **Physical Property Classes**: Represent quantities like time, pressure, mass, and dimensionless numbers with built-in unit support.
- **Unit Conversions**: Seamlessly convert between units (e.g., seconds to hours, bar to psi) using the `UnitConverter` class.
- **Array Support**: Handle scalar or NumPy array values with bounds checking and clipping.
- **Engineering Calculations**: Compute properties like surface tension from parachor or Nusselt number from correlations.
- **Plotting**: Visualize data with Plotly integration.
- **Extensible**: Add custom properties or units with a modular design.

## Installation

### Prerequisites

- Python 3.8 or higher
- `pip` (Python package manager)

### Install from PyPI

```bash
pip install physical_properties
```

### Install from Source

Clone the repository and install locally:

```bash
git clone https://github.com/ENNOVA-LLC/physical_properties.git
cd physical_properties
pip install -e .
```

### Optional Dependencies

For testing or development:

```bash
pip install "physical_properties[test]"  # For pytest
pip install "physical_properties[dev]"   # For linting/formatting tools
```

## Usage

### Basic Example

Create a `Time` object and convert units:

```python
from physical_properties import Time

# Create a time object in seconds
t = Time(value=3600, unit="s")
print(t)  # Time(name='time', unit='s', value=(3600))

# Convert to hours
t_hours = t.to("h")
print(t_hours)  # Time(name='time', unit='h', value=(1))
```

### Working with Arrays

Handle multiple values with bounds:

```python
from physical_properties import Temperature

# Temperature array with bounds
temp = Temperature(value=[300, 350, 400], unit="K", bounds=(0, 500))
print(temp.mean())  # 350.0

# Convert to Celsius
temp_c = temp.to("C")
print(temp_c)  # Temperature(name='temperature', unit='C', value=(3,))
```

### Dimensionless Numbers

Calculate a Reynolds number:

```python
from physical_properties import Length, Velocity, Density, Viscosity, ReynoldsNumber

length = Length(value=0.1, unit="m")
velocity = Velocity(value=2.0, unit="m/s")
density = Density(value=1000, unit="kg/m3")
viscosity = Viscosity(value=0.001, unit="Pa*s")

re = ReynoldsNumber.from_definition(length, velocity, density, viscosity)
print(re)  # ReynoldsNumber(name='Reynolds number', unit='None', value=(1,))
```

### Plotting

Visualize data:

```python
import numpy as np
from physical_properties import Time, Pressure

time = Time(value=np.linspace(0, 10, 100), unit="s")
pressure = Pressure(value=np.sin(time.value), unit="bar")

fig = pressure.plot(x=time)
fig.show()
```

## Package Structure

```bash
physical_properties/
├── src/
│   └── physical_properties/
│       ├── base.py              # PhysicalProperty base class
│       ├── properties/          # Property-specific modules
│       │   ├── basic.py         # Time, Length, etc.
│       │   ├── thermodynamic.py # Temperature, Pressure, etc.
│       │   ├── dimensionless.py # ReynoldsNumber, etc.
│       │   └── ...              # Other grouped or standalone properties
│       ├── units/               # Unit conversion logic
│       └── utils/               # Utility functions
├── tests/                       # Unit tests
├── examples/                    # Usage examples
└── pyproject.toml               # Package configuration
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or support, open an issue on GitHub or contact Caleb Sisco (mailto:sisco@ennova.us).
