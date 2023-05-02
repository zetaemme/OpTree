# OpTree: (Op)timal Decision (Tree)

Implementation of the DecTree algorithm presented in the paper from Cicalese, Later, Saettler.

## Author

Mattia Zorzan - *VR464472*

## References

F. Cicalese, E. Laber, A. Saettler "Decision Trees for Function Evaluation: Simultaneous Optimization of Worst and
Expected Cost", 2016: [[pdf]](https://link.springer.com/content/pdf/10.1007/s00453-016-0225-9.pdf)

## Usage

Simply run the `main.py` script. It accepts various arguments:

1. _-f_ (filename, **required**): The path to the CSV containing the dataset.
2. _-p_ (pairs): The pairs set for the given dataset can be pre-computed using the `compute_pairs.py` script. It
   generates a JSON file accepted by the main script.
