# ChangeLog

All notable changes to this project are documented in this file.

## [0.3.0] - 2025-04-04

### Changed

- Improved plotting in `PhysicalProperty` and `XYData` classes.
- Reverts `Composition` class to its original implementation. The `composition` module now uses `ChemicalComposition` instead.
- Improved serialization of `SolventBlendSpec` class and other classes in the `composition` module.

## [0.2.0] - 2025-03-27

### Added

- `properties/specialty/composition` to handle blending functionality.

### Changed

- Temporary removal of unit validation in `properties` module. Need to add more unit validation before exposing this functionality.

## [0.1.0] - 2025-03-27

### Added

- `src/physical_property` with submodules ...
  - `properties` module for defining and working with physical properties.
  - `units` module for handling unit conversion.
  - `xy_data` module for storing and handling XY data in a standard format. Also facilitates plotting.
