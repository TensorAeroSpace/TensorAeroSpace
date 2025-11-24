# XFLR5-based UAV examples

This directory contains two reference projects (folders `uav1` and `uav2`) built in XFLR5 and exported to MATLAB/Simulink for further integration with TensorAeroSpace.

## Folder structure

- `uav*/uav*.xfl` – XFLR5 project files with aerodynamic definitions.
- `uav*/uav*_model.slx` – Simulink schemes that reproduce the corresponding state-space models.
- `uav*/main.m` – MATLAB script that prepares matrices `A`, `B`, `C`, `D` and exports them to Python.
- `uav*/model.so` / `model.dll` – compiled shared libraries for Linux/Windows created with Embedded Coder.
- `uav*/rtwtypes.py` – helper with ctypes definitions used in Python bindings.

## Quick start

1. Open the `.xfl` model in XFLR5 to inspect aerodynamic assumptions (airfoils, incidence, inertia).
2. Run `main.m` from MATLAB to regenerate state-space matrices if the geometry was changed.
3. Build the Simulink model (`Ctrl+B`) to update the shared library targets. Both DLL and SO variants will be produced.
4. From Python, load the generated library with `tensoraerospace.aerospacemodel.utils.rtwtypes` helpers. Example scripts are provided in `uav*/script.py`.

Additional screenshots and a detailed tutorial are available in the documentation: `docs/en/model/uav.md` (EN) and `docs/ru/model/uav.md` (RU).



