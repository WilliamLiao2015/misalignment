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
```

### Set up LCTGen

Follow the installation instructions of [LCTGen](https://github.com/Ariostgx/lctgen). (Note: You might need to remove the version of `tensorflow` dependency from the `requirements.txt` file.)

Remember to clone the repositories to the root directory of this repository, as shown below:

```
misalignment/
└── lctgen/
```

Then install the following dependencies:

```bash
pip install openai --upgrade
pip install pydantic
pip install python-dotenv
```

### Set up LLM Service

Create `.env` or `.env.local` file in the root directory of the repository with the following content:

```
LLM_BASE_URL=<YOUR_LLM_BASE_URL>
LLM_MODEL=<YOUR_LLM_MODEL>
LLM_API_KEY=<YOUR_LLM_API_KEY>
```

Notice that this repository uses OpenAI compatible API for the LLM service.

## Usage

### Evaluate Misalignment

```bash
python benchmark.py --num_configs 1
```
