#!/bin/bash

# Navigate to the parent directory of bash_scripts
cd ..
cd Confidence_Estimation

# Run script1.py
echo "Training"
grun python Main/Train/main.py

# Run script2.py
echo "Data generation"
grun python Data/Data_generation/main.py

# Run script3.py
echo "Validation"
grun python Main/Validation/main.py

# Run script4.py
echo "Testing"
grun python Main/Test/main.py


# Return to the original directory (bash_scripts)
cd bash_scripts
