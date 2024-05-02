#!/bin/bash
# Find all Jupyter notebooks starting with 'simulation_' and execute them
for notebook in simulation_*.ipynb; do
    echo "Running $notebook..."
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=myenv "$notebook"
    echo "$notebook completed."
done

