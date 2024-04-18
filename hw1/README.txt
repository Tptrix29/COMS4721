Package Dependency:
- re
- numpy
- collections

File Structure:
- `hw1.py`: script for model class and utility functions
- `hw1-run.py`: script for running the model and generating the output

Reproducing the result: (`hw1-run.py`)
0. Install the package dependency
1. Open terminal, run `python3 hw1-run.py -h` for usage help
2. Run `python3 hw1-run.py --human path/to/human/corpus --gpt path/to/gpt/corpus` script to get the result with default parameters
3. Running log and result content will be output to the console
4. Customize the parameter to get different result

Index for functions in `hw1.py`:
- Data Preprocessing: `read_corpus` function
- Train & Test Split: `train_test_split` function
- OOV Calculation: `oov_rate_n_gram` function
- N-gram Model: `NGramModel` class
- Accuracy Evaluation: `accuracy` function
- Classification: `NGramModel.predict` function
- Text Generation: `NGramModel.generate_sentence` function
