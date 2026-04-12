# SimInTech to Python

Integrate SimInTech models into Python by running the project from a script and exchanging data through files. Below is a quick start guide and a working example.

## Quick start

1. Prepare a SimInTech project (`.prt`/`.xprt`) with the following blocks:
   - reading the input signal from a file (e.g., `sit_in_1.dat`),
   - writing the output signal to a file (e.g., `sit_out_1.dat`).
2. Find the path to the SimInTech executable `mmain.exe` (usually `C:\\SimInTech64\\bin\\mmain.exe`).
3. Generate the input signal in Python and save it to `sit_in_1.dat`.
4. Run the SimInTech project from Python: `mmain.exe <path_to_project> /run /exitonstop`.
5. Read the output file in Python and plot the results.

!!! note "Supported platform"
    The examples below are designed for Windows, since SimInTech is a Windows application.

---

## Full Python example

```python
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensoraerospace.utils import generate_time_period, convert_tp_to_sec_tp
from tensoraerospace.signals.standart import unit_step


def run_simintech(sit_bin: Path, project: Path, extra_args: list[str] | None = None, timeout_sec: int | None = 120) -> None:
    """Run a SimInTech project with the /run /exitonstop flags.

    - sit_bin: path to mmain.exe
    - project: path to .prt/.xprt
    - extra_args: additional command-line arguments
    - timeout_sec: timeout for completion, in seconds
    """
    args = [str(sit_bin), str(project), "/run", "/exitonstop"]
    if extra_args:
        args.extend(extra_args)

    completed = subprocess.run(
        args,
        check=False,           # we will show a clear error below
        capture_output=True,   # capture stdout/stderr in case of errors
        text=True,
        timeout=timeout_sec,
        shell=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "SimInTech exited with an error "
            f"(code {completed.returncode}).\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


if __name__ == "__main__":
    # Paths: adjust for your installation/project
    sit_bin = Path(r"C:\\SimInTech64\\bin\\mmain.exe")
    project = Path(r".\\lsu2.xprt")  # project in the current folder

    # Generate input signal: 1 deg step at t0=0.5 s, dt=0.01 s
    dt = 0.01
    tp = generate_time_period(tn=100, dt=dt)                 # discrete ticks
    tps = convert_tp_to_sec_tp(tp, dt=dt)                     # time in seconds

    # unit_step returns an array of values (in radians when output_rad=True)
    # Form the shape [channels, time] -- here one channel
    ref = unit_step(degree=1, tp=tp, time_step=0.5, output_rad=True)
    reference_signals = np.reshape(ref, (1, -1))

    # Write the input file. SimInTech often expects values line-by-line/component-wise.
    # Check the format of your project. In the simplest case -- a single column of values.
    in_file = Path("sit_in_1.dat")
    np.savetxt(in_file, reference_signals.ravel(), fmt="%.6f")

    # Run the simulation
    run_simintech(sit_bin=sit_bin, project=project)

    # Read the result (configure the output file path and format in the SimInTech project)
    out_file = Path("sit_out_1.dat")
    if out_file.exists():
        y = np.loadtxt(out_file, dtype=float)

        # Plot the result
        plt.plot(tps, y[: len(tps)])
        plt.xlabel("t, [s]")
        plt.ylabel("y, [units]")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print(f"Warning: output file {out_file} not found. Check the SimInTech project settings.")
```

### Explanation of key lines

- `run_simintech(...)` uses `subprocess.run` without `shell=True` and with arguments as a list -- this is more reliable and safer.
- The `/run` and `/exitonstop` flags start the simulation and close the GUI after successful completion.
- Input/output via `*.dat`: use `numpy.savetxt`/`numpy.loadtxt` for a simple text format. For multiple channels, organize columns.
- The modules `generate_time_period`, `convert_tp_to_sec_tp`, and `unit_step` are part of TensorAeroSpace and simplify the creation of test signals.

---

## Alternative: running from PowerShell

```powershell
"C:\\SimInTech64\\bin\\mmain.exe" \
  "C:\\tensoraerospace\\aerospacemodel\\simintechModel\\lsu2.xprt" \
  /run /exitonstop
```

> Quotes are required if the path contains spaces.

---

## Common issues and solutions

- SimInTech does not start from Python:
  - Check the correctness of the path to `mmain.exe`.
  - Make sure the `.xprt` project file exists and is accessible.
  - Run the script from "PowerShell (x64)".
- The process exits with a nonzero code and empty output:
  - Open the project manually and check the file input/output blocks.
  - Compare the format of `sit_in_1.dat` with what the project expects (number of columns, delimiter, encoding). For binary format, use `numpy.tofile`/`fromfile`.
- Output file is not created:
  - Make sure the write path is correct and the directory exists.
  - Check directory access permissions.

!!! warning "Data format"
    The specific file format (`*.dat`) depends on the settings of your SimInTech project. If the format does not match, update the read/write logic in Python.

---

## Checklist

- [ ] The path to `mmain.exe` is correct (or accessible via an environment variable/shortcut)
- [ ] The `.prt/.xprt` project opens manually and runs correctly
- [ ] Input reading and output writing blocks are configured for files
- [ ] Python generates the input file in the required format
- [ ] Launching via `subprocess.run` completes without errors and the output file is readable
