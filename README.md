# BOA - A Multi-Objective Bayesian Optimization Program for the gem5-Aladdin SoC Simulator

## Installation

1. Create new Python 3.6 environment (e.g., with `pipenv`, `virtualenv`, or `conda`)

    ```bash
    conda create env --name boa python=3.6
    ```

2. Activate `boa` environment

    ```bash
    conda activate boa
    ```

3. Get source code

    ```bash
    git clone <url-to-repo>
    ```
   
4. Install `BOA`

    ```bash
    pip install ./boa --upgrade
    ```
    
  
## Usage

Run `boa -h` for a quick overview of functionality.

### Run

To run `BOA` a `YML` configuration file needs to be specified. 

```bash
boa --config=run.yml
```

See [this file](resources/config.yml) for an example.


