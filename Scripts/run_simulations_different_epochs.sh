#!/bin/bash

# Define arrays for MODEL_NAMEs, DATASET_NAMEs, and LOSS_NAMEs
MODEL_NAMES=("SimpleCNN") # Add other models as needed: "VGG" "ResNet18" "SimpleCNN"
DATASET_NAMES=("HAM-10000") # Add other datasets as needed: "MNIST" "CIFAR-10" "CIFAR-100" "HAM-10000"
LOSS_NAMES=("cross_entropy") # Add other loss functions as needed  "label_smoothed_crossentropy"  "mse"
LEARNING_RATE=0.005
NUMBER_OF_EPOCHS=20

H_FLIP=False
V_FLIP=False
RANDOM_CROP=False
COLOR_JITTER=False
ROTATION=False  # New Flag for rotation
USE_MIXUP=False  # Add a flag to turn on or off mixup
USE_DIFFEOMORPHISM=False  # Flag to turn on or off the diffeomorphism transformations

NOISE=True
# Grid search range for c and temperature scale
C_VALUES=$(seq 10 10 10) # c will range from 2 to 20 with steps of 2

TEMP_SCALE_VALUES=(10.0) # temperature scale values

N_Extraction=2
FREQUENCY=2
DIFFEO_NUMBER_PER_SIM=25
TOTAL_DIFFEO_NUMBER=$((FREQUENCY * DIFFEO_NUMBER_PER_SIM))
#TOTAL_DIFFEO_NUMBER=5
# Iterate over each combination of model, dataset, loss, c, and temperature scale
NUM_RUNS=5

for (( i=1; i<=NUM_RUNS; i++ ))
do
  echo "Simul number= ${i}"
  for MODEL_NAME in "${MODEL_NAMES[@]}"
  do
      for DATASET_NAME in "${DATASET_NAMES[@]}"
      do
          for LOSS_NAME in "${LOSS_NAMES[@]}"
          do
              for c in $C_VALUES
              do
                  for temp_scale in "${TEMP_SCALE_VALUES[@]}"
                  do
                      # Set VALIDATION_ACCURACY_THRESHOLD based on DATASET_NAME
                      case $DATASET_NAME in
                          "MNIST")
                              VALIDATION_ACCURACY_THRESHOLD=97
                              BATCH_FREQUENCY=200
                              ;;
                          "CIFAR-10")
                              VALIDATION_ACCURACY_THRESHOLD=79
                              BATCH_FREQUENCY=20
                              ;;
                          "CIFAR-100")
                              VALIDATION_ACCURACY_THRESHOLD=55
                              BATCH_FREQUENCY=5
                              ;;
                          "HAM-10000")
                              VALIDATION_ACCURACY_THRESHOLD=77
                              BATCH_FREQUENCY=30
                              ;;
                      esac

                      # Lowercased and combined directory name for this combination
                      DIR_NAME="${MODEL_NAME,,}_${DATASET_NAME,,}_${LOSS_NAME}_run${i}"
                      echo "Running with MODEL_NAME=${MODEL_NAME}, DATASET_NAME=${DATASET_NAME}, LOSS_NAME=${LOSS_NAME}, VALIDATION_ACCURACY_THRESHOLD=${VALIDATION_ACCURACY_THRESHOLD}, c=${c}, temp_scale=${temp_scale}, Noise=${NOISE}"

                      # Ensure directory exists
                      mkdir -p "./Networks/${DIR_NAME}"
                      mkdir -p "./Data/${DIR_NAME}"
                      mkdir -p "./Final_Data/${DIR_NAME}"

                      # Update MODEL_NAME, DATASET_NAME, VALIDATION_ACCURACY_THRESHOLD, and LOSS_NAME in General_Training.py
                      sed -i "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" General_Training_multi_save.py
                      sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" General_Training_multi_save.py
                      sed -i "s/^VALIDATION_ACCURACY_THRESHOLD = .*/VALIDATION_ACCURACY_THRESHOLD = ${VALIDATION_ACCURACY_THRESHOLD}/" General_Training_multi_save.py
                      sed -i "s/^LOSS_NAME = .*/LOSS_NAME = '${LOSS_NAME}'/" General_Training_multi_save.py
                      sed -i "s/^BATCH_FREQUENCY = .*/BATCH_FREQUENCY = ${BATCH_FREQUENCY}/" General_Training_multi_save.py
                      sed -i "s/^LEARNING_RATE = .*/LEARNING_RATE = ${LEARNING_RATE}/" General_Training_multi_save.py
                      sed -i "s/^NUMBER_OF_EPOCHS = .*/NUMBER_OF_EPOCHS = ${NUMBER_OF_EPOCHS}/" General_Training_multi_save.py
                      sed -i "s/^H_FLIP = .*/H_FLIP = ${H_FLIP}/" General_Training_multi_save.py
                      sed -i "s/^V_FLIP = .*/V_FLIP = ${V_FLIP}/" General_Training_multi_save.py
                      sed -i "s/^RANDOM_CROP = .*/RANDOM_CROP = ${RANDOM_CROP}/" General_Training_multi_save.py
                      sed -i "s/^COLOR_JITTER = .*/COLOR_JITTER = ${COLOR_JITTER}/" General_Training_multi_save.py
                      sed -i "s/^ROTATION = .*/ROTATION = ${ROTATION}/" General_Training_multi_save.py
                      sed -i "s/^USE_MIXUP = .*/USE_MIXUP = ${USE_MIXUP}/" General_Training_multi_save.py
                      sed -i "s/^NOISE = .*/NOISE = ${NOISE}/" General_Training_multi_save.py
                      sed -i "s/^c_value = .*/c_value = ${c}/" General_Training_multi_save.py
                      sed -i "s/^temperature_scale = .*/temperature_scale = ${temp_scale}/" General_Training_multi_save.py


                      # Execute the general training script with the current configuration
                      grun python General_Training_multi_save.py $MODEL_NAME $DATASET_NAME $c $temp_scale $LOSS_NAME

                      # Find all files matching the batch save pattern
                      SAVE_PATH_PATTERN="./Networks/${MODEL_NAME,,}_${DATASET_NAME,,}_batch*.pth"
                      SAVE_FILES=($(ls $SAVE_PATH_PATTERN 2> /dev/null | sort -V))

                      for file in "${SAVE_FILES[@]}"; do
                          echo "$file"
                          BATCH_NUMBER=$(echo "$file" | grep -oP '(?<=batch)\d+(?=.pth)')
                          echo "Extracted BATCH_NUMBER: $BATCH_NUMBER"
                      done
                      NUM_SAVES=${#SAVE_FILES[@]}

                      # Logarithmically sample the saves if there are more than required
                      if [[ $NUM_SAVES -gt $((N_Extraction + 1)) ]]; then

                          # Logarithmically spaced indices (1-indexed for shell array)
                          INDICES=($(python -c "import numpy as np; idx = np.unique(np.logspace(np.log10(1), np.log10($NUM_SAVES), num=${N_Extraction}, dtype=int)); print(' '.join(map(str, idx)))"))
                          echo "Keeping files with indices: ${INDICES[@]}"

                          # Delete files that are not in the INDICES
                          for ((i=1; i<=$NUM_SAVES; i++)); do
                              if ! [[ " ${INDICES[*]} " =~ " ${i} " ]]; then

                                  rm "${SAVE_FILES[$((i-1))]}"
                              fi
                          done
                          echo "Deleting all non important saves"
                          for index in "${INDICES[@]}"; do
                              # Compute 0-indexed position
                              let "pos = $index - 1"
                              TOTAL_BATCH_COUNT=$(echo "${SAVE_FILES[$pos]}" | grep -oP '(?<=batch)\d+(?=.pth)')
                              FILE_LOCATION="./Networks/${MODEL_NAME,,}_${DATASET_NAME,,}.pth"
                               #BATCH_NUMBER=$(echo "${SAVE_FILES[$pos]}" | grep -oP '(?<=batch)\d+(?=.pth)')
                              echo "Extracted batch number is : $TOTAL_BATCH_COUNT"
                              # Rename the file for use
                              mv "${SAVE_FILES[$pos]}" $FILE_LOCATION

                              # Run the generating and post hoc scripts
                              echo "Processing for c=${c}, temp_scale=${temp_scale}"

                              # Update DIFFEOMORPHISM_PARAMS in Generating_diffeo_images_General.py
                              sed -i "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" Generating_diffeo_images_General.py
                              sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" Generating_diffeo_images_General.py
                              sed -i "s/^TOTAL_DIFFEO_NUMBER = .*/TOTAL_DIFFEO_NUMBER = ${TOTAL_DIFFEO_NUMBER}/" Generating_diffeo_images_General.py
                              sed -i "/\"temperature scale\":/s/: [^,]*/: $temp_scale/" Generating_diffeo_images_General.py
                              sed -i "/\"c\":/s/: [^}]*/: $c/" Generating_diffeo_images_General.py

                              # Execute the script for generating diffeomorphic images
                              grun python Generating_diffeo_images_General.py $MODEL_NAME $DATASET_NAME

                              # Update MODEL_NAME and DATASET_NAME in General_post_hoc.py
                              sed -i "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" General_post_hoc.py
                              sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" General_post_hoc.py
                              sed -i "s/^DIFFEO_NUMBER_PER_SIM = .*/DIFFEO_NUMBER_PER_SIM = ${DIFFEO_NUMBER_PER_SIM}/" General_post_hoc.py

                              # Execute the post hoc analysis script
                              grun python General_post_hoc.py $MODEL_NAME $DATASET_NAME

                              #NAME_PATTERN="c${c}_temp${temp_scale}_batch${TOTAL_BATCH_COUNT}"
                              NAME_PATTERN="batch${TOTAL_BATCH_COUNT}_run${i}"
                              # mv the network into an appropriate folder
                              # Move and rename files according to the new variables including c and temperature scale
                              mv "./Data/${MODEL_NAME,,}_${DATASET_NAME,,}.pth" "./Data/${DIR_NAME}/data_${NAME_PATTERN}.pth"
                              mv "./Final_Data/${MODEL_NAME,,}_${DATASET_NAME,,}/results.pth" "./Final_Data/${DIR_NAME}/results_${NAME_PATTERN}.pth"
                              mv "./Final_Data/${MODEL_NAME,,}_${DATASET_NAME,,}/eval_method.pth" "./Final_Data/${DIR_NAME}/eval_method_${NAME_PATTERN}.pth"
                              mv "./Networks/${MODEL_NAME,,}_${DATASET_NAME,,}.pth" "./Networks/${DIR_NAME}/save_${NAME_PATTERN}.pth"

                          done
                      fi
                  done
              done
          done
      done
  done
done
