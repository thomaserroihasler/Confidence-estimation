#!/bin/bash

# Define arrays for MODEL_NAMEs, DATASET_NAMEs, and LOSS_NAMEs
MODEL_NAMES=("SimpleCNN") # Add other models as needed: "VGG" "ResNet18" "SimpleCNN"
DATASET_NAMES=("MNIST") # Add other datasets as needed: "MNIST" "CIFAR-10" "CIFAR-100" "HAM-10000"
LOSS_NAMES=("cross_entropy") # Add other loss functions as needed  "label_smoothed_crossentropy"  "mse"

# Grid search range for c and temperature scale
C_VALUES=$(seq 2 2 20)            # c will range from 5 to 20 with steps of 5
TEMP_SCALE_VALUES=( 0.1 0.177827941 0.316 0.56234132519 1.0 1.77827941004 3.16 5.6234132519 10.0)    # temperature scale values are 0.1, 1.0, and 10.0

# Iterate over each combination of model, dataset, and loss
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    for DATASET_NAME in "${DATASET_NAMES[@]}"
    do
        for LOSS_NAME in "${LOSS_NAMES[@]}"
        do
            # Set VALIDATION_ACCURACY_THRESHOLD based on DATASET_NAME
            case $DATASET_NAME in
                "MNIST")
                    VALIDATION_ACCURACY_THRESHOLD=95
                    ;;
                "CIFAR-10")
                    VALIDATION_ACCURACY_THRESHOLD=80
                    ;;
                "CIFAR-100")
                    VALIDATION_ACCURACY_THRESHOLD=55
                    ;;
                "HAM-10000")
                    VALIDATION_ACCURACY_THRESHOLD=60
                    ;;
            esac

            # Lowercased and combined directory name for this combination
            DIR_NAME="${MODEL_NAME,,}_${DATASET_NAME,,}_${LOSS_NAME}"
            echo "Running training for MODEL_NAME=${MODEL_NAME}, DATASET_NAME=${DATASET_NAME}, LOSS_NAME=${LOSS_NAME}, VALIDATION_ACCURACY_THRESHOLD=${VALIDATION_ACCURACY_THRESHOLD}"

            # Ensure directory exists
            mkdir -p "./Networks/${DIR_NAME}"
            mkdir -p "./Data/${DIR_NAME}"
            mkdir -p "./Final_Data/${DIR_NAME}"

            # Update MODEL_NAME, DATASET_NAME, VALIDATION_ACCURACY_THRESHOLD, and LOSS_NAME in General_Training.py
            sed -i "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" General_Training.py
            sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" General_Training.py
            sed -i "s/^VALIDATION_ACCURACY_THRESHOLD = .*/VALIDATION_ACCURACY_THRESHOLD = ${VALIDATION_ACCURACY_THRESHOLD}/" General_Training.py
            sed -i "s/^LOSS_NAME = .*/LOSS_NAME = '${LOSS_NAME}'/" General_Training.py

            # Execute the general training script with the current configuration
            grun python General_Training.py $MODEL_NAME $DATASET_NAME $LOSS_NAME

            # Network training is now complete, start processing for each c and temp_scale
            for c in $C_VALUES
            do
                for temp_scale in "${TEMP_SCALE_VALUES[@]}"
                do
                    echo "Processing for c=${c}, temp_scale=${temp_scale}"

                    # Update DIFFEOMORPHISM_PARAMS in Generating_diffeo_images_General.py
                    sed -i "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" Generating_diffeo_images_General.py
                    sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" Generating_diffeo_images_General.py
                    sed -i "/\"temperature scale\":/s/: [^,]*/: $temp_scale/" Generating_diffeo_images_General.py
                    sed -i "/\"c\":/s/: [^}]*/: $c/" Generating_diffeo_images_General.py

                    # Execute the script for generating diffeomorphic images
                    grun python Generating_diffeo_images_General.py $MODEL_NAME $DATASET_NAME

                    # Update MODEL_NAME and DATASET_NAME in General_post_hoc.py
                    sed -i "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" General_post_hoc.py
                    sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" General_post_hoc.py

                    # Execute the post hoc analysis script
                    grun python General_post_hoc.py $MODEL_NAME $DATASET_NAME

                    # Define a specific name pattern including the c value and temperature scale
                    NAME_PATTERN="c${c}_temp${temp_scale}"

                    # Move and rename files according to the new variables including c and temperature scale
                    mv "./Data/${MODEL_NAME,,}_${DATASET_NAME,,}.pth" "./Data/${DIR_NAME}/data_${NAME_PATTERN}.pth"
                    mv "./Final_Data/${MODEL_NAME,,}_${DATASET_NAME,,}/results.pth" "./Final_Data/${DIR_NAME}/results_${NAME_PATTERN}.pth"
                    mv "./Final_Data/${MODEL_NAME,,}_${DATASET_NAME,,}/eval_method.pth" "./Final_Data/${DIR_NAME}/eval_method_${NAME_PATTERN}.pth"
                done
            done
          # Move the trained model to the appropriate directory after processing all c and temp_scale values
            mv "./Networks/${MODEL_NAME,,}_${DATASET_NAME,,}.pth" "./Networks/${DIR_NAME}/save_${MODEL_NAME,,}_${DATASET_NAME,,}.pth"
        done
    done
done
