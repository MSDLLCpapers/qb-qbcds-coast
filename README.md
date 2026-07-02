# Causal Intelligence for Constraint-Aware Intervention Design to Induce State Transitions

This repository contains the code for COAST (Causally Optimal Actions for State Transitions), a causal-intelligence approach for the in-silico design of constrained interventions that induce user-defined state transitions.

<img width="1612" src="figures/DiagramCOAST.jpg">


## Getting Started

COAST has three external runtime dependencies that you must install yourself
before installing the Python package:

1. **A Java Development Kit (JDK 17+)** &mdash; required by JPype1 to invoke
   the Tetrad causal-discovery library. OpenJDK is recommended.
2. **py-tetrad** (the Tetrad Java library + Python wrapper) &mdash; required
   at runtime for causal discovery. Not bundled with COAST.
3. **An R interpreter (R &ge; 4.0) with rpy2 and GENIE3** &mdash;
   required only for the feature selection step, can be optional.

### 1. Install the JDK

```bash
# HPC with environment modules
module load java/openjdk-17

# Ubuntu/Debian
sudo apt install openjdk-17-jdk

# macOS (Homebrew)
brew install openjdk@17
```

Verify:

```bash
java -version       # should report 17 or later
echo $JAVA_HOME     # should point at the JDK install directory
```

### 2. Install py-tetrad

py-tetrad runs [Tetrad](https://www.cmu.edu/dietrich/philosophy/tetrad/) in Python. COAST does **not** bundle it; py-tetrad must be installed separately:

```bash
pip install git+https://github.com/cmu-phil/py-tetrad
git clone https://github.com/cmu-phil/py-tetrad.git
# Note the absolute path; you will need it in step 4.
```

### 3. Create a Python environment and install COAST

**Option A &mdash; conda (recommended):**

```bash
conda env create -f environment.yml
conda activate causal
pip install -e .          # install COAST itself in editable mode
```

**Option B &mdash; pure pip / venv:**

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -e .          # also pulls everything from pyproject.toml
```

Optional extras:

```bash
pip install -e ".[notebooks]"     # JupyterLab + ipykernel
pip install -e ".[r]"             # rpy2 (GPLv2+, opt-in)
pip install -e ".[notebooks,r]"   # both
```

### 4. Tell COAST where py-tetrad lives

COAST loads the Tetrad jar through JPype using the path to your local
py-tetrad checkout. Set the environment variable `PYTETRAD_HOME` to the
absolute path:

```bash
export PYTETRAD_HOME=/path/to/py-tetrad
```

The example notebooks under `results/` read this variable when starting
the JVM with the Tetrad jar on the classpath.

### 5. Install R and GENIE3 for the feature selection step *(Optional)*

If you want to use `coast.feature_selection.run_genie3` (which calls into
R via rpy2), install R, the rpy2 extra, and the required R packages:

```bash
module load R/4.3        # (HPC) or install R from https://www.r-project.org/
pip install -e ".[r]"
```

Then in an R session:

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("GENIE3")
install.packages("doRNG")
install.packages("doParallel")
```

See `GENIE3_via_rpy2.txt` for the full setup procedure on HPC.

### 6. Verify the installation

```bash
python -c "import coast; print(coast.__file__)"
python -c "from coast.optimization import CausalClass; print('ok')"
```

If both succeed, COAST is installed.

## Repository Contents

- `coast` contains the source code of COAST
- `results` contains the experiment results of COAST:
  - `synthetic` contains the results on synthetic datasets
  - `perturb-CITE-seq` contains the results on a Perturb-CITE-seq dataset from pooled CRISPR perturbations
  - `chen-scRNA-seq` contains the results on a real single-cell RNA-seq dataset of adult mouse hypothalamus
- `figures` contains the figures generated for publication

## Licenses

- COAST itself is released under the Apache License, Version 2.0
  (see [`LICENSE`](LICENSE)).
- py-tetrad / Tetrad are GPLv2 and are **not** bundled. They live on the
  user's machine after step 2.
- rpy2, GENIE3, doRNG, and doParallel are GPL-family and are **not**
  installed by default.
- See [`LICENSES_THIRD_PARTY`](LICENSES_THIRD_PARTY) for the full
  third-party license breakdown.

## Citation

@misc{song2026causalintelligenceconstraintawareintervention,\
  title={Causal Intelligence for Constraint-Aware Intervention Design to Induce State Transitions},\
  author={Zixuan Song and Uwe Mueller and Dimitris V. Manatakis},\
  year={2026},\
  eprint={2605.29008},\
  archivePrefix={arXiv},\
  primaryClass={cs.LG},\
  url={https://arxiv.org/abs/2605.29008 },  
}

## Contact

For any questions, please submit a GitHub issue or contact [Zixuan Song](zixuan.song@msd.com) or [Dimitris  V. Manatakis](dimitrios.manatakis@msd.com).
