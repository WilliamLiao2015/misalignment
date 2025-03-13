# Misalignment Evaluation for Vehicle Trajectory Generation Methods

## Installation

### Clone the Repository

```bash
git clone https://github.com/WilliamLiao2015/misalignment.git
```

### Install Dependencies

```bash
conda create -n misalignment python=3.8
conda activate misalignment
cd misalignment
pip install -r requirements.txt
```

### Set up the Trajectory Generation Methods

Follow the installation instructions of supported trajectory generation methods to set up the environments.

Create a new conda environment for each of the following trajectory generation methods:

- [LCTGen](https://github.com/Ariostgx/lctgen)
    - Note: You might need to remove the version of `tensorflow` dependency from the `requirements.txt` file.
- [ProSim](https://github.com/Ariostgx/ProSim)

Remember to clone the repositories to the root directory of this repository, as shown below:

```
misalignment/
├── lctgen/
├── ProSim/
└── ...
```

For each of the trajectory generation methods, install the following dependencies:

```bash
conda activate <TRAJECTORY_GENERATION_METHOD_ENV>
pip install openai --upgrade
pip install pydantic
pip install python-dotenv
```

### Set up the LLM Service.

Create `.env` file in the root directory of the repository with the following content:

```
LLM_BASE_URL=<YOUR_LLM_BASE_URL>
LLM_MODEL=<YOUR_LLM_MODEL>
LLM_API_KEY=<YOUR_LLM_API_KEY>
```

Notice that this repository uses OpenAI compatible API for the LLM service.

## Usage

### Evaluate Misalignment

```bash
python -m test --config <CONFIG_FILE> --method <METHOD>
```
