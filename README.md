# Grocery store scheduling

Assignment 2 for ECSE526 AI course at McGill University

## Launch with default parameters
Install the following libraries to ensure the program works :
```
pip3 install numpy scipy
```

## Launch with default parameters

To launch the algorithms using default parameters, use the following command :
```bash
python3 main.py --algo <name_of_algo>
```
"<name_of_algo>" with either V or P, respectively for value iteration or policy iteration.

## Launch with specific parameters

To know all the parameters that can be changed, I recommend using the -h flag :
```bash
python3 main.py -h
```

Example of use with some personal parameters :
```bash
python3 main.py --algo V --N 15 --M 40 --P 0.6 --customer_law poisson
```