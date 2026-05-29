# Phase 7.1: PyBullet Setup Spike

Phase 7.1 adds the smallest useful PyBullet setup for local physics simulation.

This spike verifies that the project can start a PyBullet physics client, create a simple plane, step the simulation, and disconnect cleanly on Windows. It does not connect physics to the existing grid-world, RL, MARL, or communication systems.

## What PyBullet Is Used For

PyBullet is used as the first lightweight physics backend for Phase 7.

For this setup spike, PyBullet is only responsible for:

- Opening a physics simulation connection.
- Creating a static ground plane.
- Advancing the simulation for a few steps.
- Disconnecting cleanly.

The new code lives under `marlsim.physics` so it stays separate from `GridWorldEnv` and the earlier grid-world phases.

## Optional Dependency

PyBullet is documented as an optional dependency:

```powershell
python -m pip install -e ".[physics]"
```

The base project still has no required runtime dependencies. Existing grid-world demos and tests should continue to work without installing PyBullet.

On this Windows Python 3.11 setup, pip tried to build PyBullet from source and failed because Microsoft C++ Build Tools were not installed.

Investigation notes:

- Current local Python: 3.11.9 on Windows.
- PyPI release metadata for recent PyBullet versions shows Linux wheels for CPython 3.9, 3.10, and 3.11.
- PyPI release metadata does not show modern Windows wheels for CPython 3.9, 3.10, or 3.11.
- Switching from Python 3.11 to a normal pip-based Python 3.10 or 3.9 virtual environment is therefore not expected to avoid a source build by itself.
- Conda-forge does publish win-64 PyBullet packages, so a Conda or Mamba environment is the cleanest source-build-free Windows path if that tooling is available.

The physics helper and tests remain import-safe when PyBullet is not installed.

## Windows Setup Options

## Recommended Windows Workflow

Use a separate Conda or Mamba environment for Phase 7 physics work. PyBullet remains optional and isolated from the base project.

Conda:

```powershell
conda create -n marlsim-physics python=3.11
conda activate marlsim-physics
conda install -c conda-forge pybullet
python -m pip install -e .
python -m marlsim.demos.physics_setup_demo
```

Mamba:

```powershell
mamba create -n marlsim-physics python=3.11 pybullet -c conda-forge
conda activate marlsim-physics
python -m pip install -e .
python -m marlsim.demos.physics_setup_demo
```

Conda-forge provides Windows `win-64` builds for PyBullet. This avoids the pip source-build path on Windows.

Pip may try to build PyBullet from source on Windows when no compatible wheel is available. Source builds may require Microsoft C++ Build Tools.

### Option A: Install Microsoft C++ Build Tools

Use this path if you want to keep using the current Python 3.11 pip workflow.

1. Install Microsoft C++ Build Tools from Microsoft.
2. Include the C++ build workload and Windows SDK.
3. Restart the terminal.
4. Install the optional dependency:

```powershell
python -m pip install -e ".[physics]"
```

This allows pip to compile PyBullet from source when no compatible Windows wheel is available.

Tradeoff: this is the most direct path for the current environment, but it installs a native compiler toolchain and may take time to build PyBullet.

### Option B: Use A Python 3.10 Virtual Environment

Use this path if you want to isolate physics work from the main Python 3.11 environment.

```powershell
py -3.10 -m venv .venv-pybullet
.\.venv-pybullet\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[physics]"
```

Important: based on current PyPI release metadata, a pip-based Python 3.10 environment on Windows may still build PyBullet from source. Python 3.10 is useful for isolation, but it is not guaranteed to avoid Microsoft C++ Build Tools.

Before relying on this path, check whether a binary wheel is available:

```powershell
python -m pip install --only-binary=:all: pybullet
```

If this reports no matching distribution, pip does not have a compatible Windows wheel for that interpreter.

### Option C: Keep Physics Optional And Skip Physics Engine Tests

Use this path if you want to continue grid-world, RL, MARL, and documentation work without installing PyBullet yet.

The repository is designed so this works:

- PyBullet is only an optional `physics` extra.
- `marlsim.physics` imports without importing PyBullet immediately.
- PyBullet-specific tests are skipped when PyBullet is unavailable.
- Existing grid-world tests continue to run.

Run the normal test suite:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover tests
```

This is the safest fallback until the physics dependency is installed cleanly.

## Why DIRECT Mode Is Used First

The setup demo starts PyBullet in `DIRECT` mode.

`DIRECT` mode runs the physics engine without opening a graphical window. This makes it better for:

- Quick setup validation.
- Automated tests.
- Headless environments.
- Windows compatibility checks before adding visual demos.

Graphical PyBullet mode can be considered later, after the basic physics helper is stable.

## How To Run The Setup Demo

From the repository root:

```powershell
$env:PYTHONPATH="src"
python -m marlsim.demos.physics_setup_demo
```

Expected output:

```text
PyBullet DIRECT setup succeeded.
Created plane body id: 0
Ran 10 simulation steps and disconnected cleanly.
```

The exact plane body id may vary.

## What Is Included

Phase 7.1 includes:

- `PhysicsWorld`, a small PyBullet helper.
- DIRECT-mode PyBullet connection support.
- Plane creation.
- Simulation stepping.
- Clean disconnect behavior.
- A minimal setup demo.
- Lightweight tests that skip PyBullet-specific execution when the optional dependency is not installed.

## What Is Intentionally Not Included Yet

Phase 7.1 does not include:

- Robot bodies.
- Obstacles.
- Goals.
- Sensors.
- RL integration.
- MARL integration.
- Communication integration.
- Continuous-control learning.
- Deep RL.
- ROS.
- Gazebo.
- Real robot deployment.
- Graphical demo requirements.

Those topics belong to later Phase 7 subphases after the local PyBullet setup is proven.

## Next Step

Before starting Phase 7.2, stabilize Phase 7.1 by choosing one Windows setup path and confirming the setup demo runs successfully.

After PyBullet is installed and the setup demo passes, the next milestone is Phase 7.2: single robot body. That step should add one simple robot shape with basic position, orientation, and movement commands while keeping the physics system isolated from the existing grid-world logic.
