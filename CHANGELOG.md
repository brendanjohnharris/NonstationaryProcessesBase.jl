# Changelog

## [0.3.0]

### Changed
- Migrated time-series backend from `TimeseriesTools` to `TimeseriesBase`.
- Renamed the `TimeSeries` constructor/methods to `Timeseries`/`timeseries`; `Process` solutions are now built via `TimeseriesBase.Timeseries`.
- Variable dimension switched from the `:Variable` symbol to the `Var` type.
- `Process.alg` type widened from `SciMLAlgorithm` to `AbstractSciMLAlgorithm`.
- Bumped `DimensionalData` compat.
- CI: added scheduled and manual (`workflow_dispatch`) triggers.

### Added
- Generic solve interface declared as stubs in `Process.jl` (`dsolve`, `odeproblem`, `process2problem`, `process2solution`), extendable by downstream simulation packages.
- `process2ds`, building a `DynamicalSystem` from a `Process` (in `DynamicalSystems.jl`).
- `Base.Dict(::Process; trim)`; the resulting metadata is attached to output `ToolsArray`s.

### Fixed
- Unstable/short solutions are now NaN-padded to the expected length instead of erroring or returning truncated series.

### Removed
- `DifferentialEquations` `Requires` hook and `src/DifferentialEquations.jl` (solve interface moved into `Process.jl`).

## [0.2.3] - 2024-11-26

### Changed
- Updated TimeseriesTools and `DimensionalData` compat; minor `Process.jl` fix.
- Updated CI and README.

## [0.2.2] - 2024-10-28

### Added
- `src/DifferentialEquations.jl` (loaded via `Requires`), exporting `dsolve`, `process2problem`, `process2solution`, `odeproblem`, and `process2ds`.
- Dependabot configuration.

### Changed
- Cleaned up `ParameterProfiles` exports.

## [0.2.1] - 2024-10-16

### Changed
- Backend array types moved from `DimArray`/`Ti` to `ToolsArray`/`𝑡`.
- Updated for newer package versions; removed `DifferentialEquations` from tests; updated docs Julia version.

### Fixed
- Replaced the `tuplef2ftuple` type handling with a new `flat_tuple` helper.
- Parameter update/profile bugs.

## [0.2.0] - 2023-11-24

### Changed
- Migrated time-series handling to `TimeseriesTools` (`TimeSeries`/`times` namespaced; arrays become `DimArray`-backed).
- Restructured package layout; moved `PyPlotTools` into `src/Plots/`.
- Default cap on number of points plotted.

### Added
- Exports `process_aliases`, `fieldguide`, and `trimtransient!`.
- Test suite.

### Fixed
- `tuplef2ftuple` type-case fixes; parameter update bug.

## [0.1.0] - 2022-06-28

- Initial release.

[0.3.0]: https://github.com/brendanjohnharris/NonstationaryProcessesBase/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/brendanjohnharris/NonstationaryProcessesBase/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/brendanjohnharris/NonstationaryProcessesBase/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/brendanjohnharris/NonstationaryProcessesBase/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/brendanjohnharris/NonstationaryProcessesBase/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/brendanjohnharris/NonstationaryProcessesBase/releases/tag/v0.1.0
