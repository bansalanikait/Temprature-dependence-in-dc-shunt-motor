# DC Shunt Motor Electro-Thermal Simulator

This project implements a Python + Streamlit application for the seven-equation DC shunt/separately excited motor electro-thermal model described in the project specification. It includes numerical integration with SciPy `solve_ivp`, validation checks, interactive Plotly charts, parameter presets, and CSV export.

## Features

- Full seven-state differential model for field current, armature current, speed, and four temperatures.
- Central parameter configuration with engineering metadata and sources.
- Input panel for supply voltage, load torque, ambient temperature, initial conditions, and simulation settings.
- Live dashboard showing operating indicators and thermal values.
- Interactive graphs for current, temperature, torque, and power.
- Live plots for current, speed, all four temperatures, torque, power, and the whole-motor balance residual.
- Thermal and whole-motor balance checks report the worst absolute residual over the sampled run.
- CSV export for recorded simulation runs and a complete ZIP report with the HTML charts, CSV data, exact test conditions, and configured parameter values.

## Installation

1. Create a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Launch the application from the repository root:

   ```bash
   streamlit run app.py
   ```

   The package entry point also works when launching from inside `dc_motor_simulator`:

   ```bash
   cd dc_motor_simulator
   streamlit run app.py
   ```

4. Or run the package directly from the repository root:

   ```bash
   python -m dc_motor_simulator.app
   ```

## Notes

- The model intentionally keeps the supplied equations and does not replace them with a simplified equivalent-circuit model.
- Heat-loss fractions can be automatically recomputed from the resistance split or edited explicitly with validation warnings when they are inconsistent.
- Parameter values are labelled as measured, manufacturer-provided, literature-derived, calculated, or estimated and are centrally configurable.
