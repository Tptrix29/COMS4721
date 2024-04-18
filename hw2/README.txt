Package Dependency:
- numpy
- matplotlib
- scipy

File Structure:
- `nn.py`: script for neural network model and related classes
- `hw2.py`: script for utility functions
- `hw2-run.py`: script for running the model and generating the output

Reproducing the result: (`hw2-run.py`)
0. Install the package dependency
1. Open terminal, run `python hw2-run.py -h` for usage help
2. Run `python hw2-run.py --file path/to/picture/file -o ./figs --optim SGD --lr 1E-3 -i 1` script to get the result with default parameters (Remember to change '--file' parameter)
3. Running log will be output to the console
4. Customize the parameter to get different result

Index for class in `nn.py`:
1. Generic Class:
    - Layer class: `Module` class
    - Loss class: `Loss` class
    - Optimizer class: `Optimizer` class
    - Parameter class: `Parameter` class
2. Concrete Class:
    - Linear layer class: `Linear` class
    - Sigmoid layer class: `Sigmoid` class
    - MeanErrorLoss loss class: `MeanErrorLoss` class
    - SGD optimizer class: `SGD` class
    - Adam optimizer class: `Adam` class
3. Other:
    - Data loader utils class: `DataLoader` class
    - Fully-connected neural network: `NeuralNetwork` class

Index for functions in `hw2.py`:
    - Image data conversion: `convert_img` function
    - Neural network training: `train` function
