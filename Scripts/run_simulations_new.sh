##!/bin/bash
#
## Define arrays for MODEL_NAMEs and DATASET_NAMEs
#MODEL_NAMES=( "SimpleCNN" "VGG" "ResNet18")
#DATASET_NAMES=("MNIST" "CIFAR-10" "CIFAR-100" "HAM-10000")
#
#NOISE=True
#
#c=5
#temp_scale=1.
#
#N_Extraction=20
#FREQUENCY=1
#DIFFEO_NUMBER_PER_SIM=30
#TOTAL_DIFFEO_NUMBER=$((FREQUENCY * DIFFEO_NUMBER_PER_SIM))
#
## Associative array to define the validation accuracy threshold for each dataset
#
#declare -A ACCURACY_THRESHOLDS
#ACCURACY_THRESHOLDS["MNIST"]=95
#ACCURACY_THRESHOLDS["CIFAR-10"]=70
#ACCURACY_THRESHOLDS["CIFAR-100"]=55
#ACCURACY_THRESHOLDS["HAM-10000"]=60
#
## Iterate over each combination of model and dataset
#for MODEL_NAME in "${MODEL_NAMES[@]}"
#do
#    for DATASET_NAME in "${DATASET_NAMES[@]}"
#    do
#        # Get the accuracy threshold for the current dataset
#        THRESHOLD=${ACCURACY_THRESHOLDS[$DATASET_NAME]}
#
#        # Lowercased and combined directory name
#        DIR_NAME="${MODEL_NAME,,}_${DATASET_NAME,,}"
#
#        # Ensure directory exists
#        mkdir -p "./Networks/${DIR_NAME}"
#        mkdir -p "./Data/${DIR_NAME}"
#        mkdir -p "./Final_Data/${DIR_NAME}"
#
#        # Update MODEL_NAME, DATASET_NAME, and VALIDATION_ACCURACY_THRESHOLD in Python files
#        sed -i "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" General_Training.py
#        sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" General_Training.py
#        sed -i "s/^VALIDATION_ACCURACY_THRESHOLD = .*/VALIDATION_ACCURACY_THRESHOLD = ${THRESHOLD}/" General_Training.py
#
#        sed -i "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" Generating_diffeo_images_General.py
#        sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" Generating_diffeo_images_General.py
#        sed -i "s/^NOISE = .*/NOISE = ${NOISE}/" Generating_diffeo_images_General.py
#        sed -i "s/^TOTAL_DIFFEO_NUMBER = .*/TOTAL_DIFFEO_NUMBER = ${TOTAL_DIFFEO_NUMBER}/" Generating_diffeo_images_General.py
#        sed -i "/\"temperature scale\":/s/: [^,]*/: $temp_scale/" Generating_diffeo_images_General.py
#        sed -i "/\"c\":/s/: [^}]*/: $c/" Generating_diffeo_images_General.py
#
#        sed -i "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" General_post_hoc.py
#        sed -i "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" General_post_hoc.py
#        sed -i "s/^DIFFEO_NUMBER_PER_SIM = .*/DIFFEO_NUMBER_PER_SIM = ${DIFFEO_NUMBER_PER_SIM}/" General_post_hoc.py
#
#        # Run Python scripts with updated variables
#        grun python General_Training.py
#        grun python Generating_diffeo_images_General.py $MODEL_NAME $DATASET_NAME
#        grun python General_post_hoc.py $MODEL_NAME $DATASET_NAME
#
#        # Move and rename files according to the new variables
#        # This is an example; you will need to ensure these paths are correct for your specific scenario
#        mv "./Networks/${MODEL_NAME,,}_${DATASET_NAME,,}.pth" "./Networks/${DIR_NAME}/save_${MODEL_NAME}_${DATASET_NAME}.pth"
#        mv "./Data/${MODEL_NAME,,}_${DATASET_NAME,,}.pth" "./Data/${DIR_NAME}/data_${MODEL_NAME}_${DATASET_NAME}.pth"
#        mv "./Final_Data/${MODEL_NAME,,}_${DATASET_NAME,,}/results.pth" "./Final_Data/${DIR_NAME}/results_${MODEL_NAME}_${DATASET_NAME}.pth"
#        mv "./Final_Data/${MODEL_NAME,,}_${DATASET_NAME,,}/eval_method.pth" "./Final_Data/${DIR_NAME}/eval_method_${MODEL_NAME}_${DATASET_NAME}.pth"
#
#    done
#done


##!/bin/bash
#
## Define arrays for MODEL_NAMEs and DATASET_NAMEs
#MODEL_NAMES=( "SimpleCNN" "VGG" "ResNet18" )
#DATASET_NAMES=( "MNIST" "CIFAR-10" "CIFAR-100" "HAM-10000" )
#
#NOISE=True
#c=5
#temp_scale=1.
#N_Extraction=20
#FREQUENCY=1
#DIFFEO_NUMBER_PER_SIM=30
#TOTAL_DIFFEO_NUMBER=$(( FREQUENCY * DIFFEO_NUMBER_PER_SIM ))
#
## Declare accuracy thresholds for each dataset
#THRESHOLD_MNIST=95
#THRESHOLD_CIFAR_10=70
#THRESHOLD_CIFAR_100=55
#THRESHOLD_HAM_10000=60
#
## Function to get the accuracy threshold based on dataset name
#get_accuracy_threshold() {
#  local dataset_name=$1
#  case "$dataset_name" in
#    "MNIST")
#      echo "$THRESHOLD_MNIST"
#      ;;
#    "CIFAR-10")
#      echo "$THRESHOLD_CIFAR_10"
#      ;;
#    "CIFAR-100")
#      echo "$THRESHOLD_CIFAR_100"
#      ;;
#    "HAM-10000")
#      echo "$THRESHOLD_HAM_10000"
#      ;;
#    *)
#      echo "Error: No threshold set for dataset $dataset_name." >&2
#      exit 1
#      ;;
#  esac
#}
#
## Iterate over each combination of model and dataset
#for MODEL_NAME in "${MODEL_NAMES[@]}"; do
#    for DATASET_NAME in "${DATASET_NAMES[@]}"; do
#
#        # Convert model and dataset names to lowercase
#        MODEL_NAME_LOWER=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
#        DATASET_NAME_LOWER=$(echo "$DATASET_NAME" | tr '[:upper:]' '[:lower:]')
#
#        # Get the accuracy threshold for the current dataset
#        THRESHOLD=$(get_accuracy_threshold "$DATASET_NAME")
#
#        # Lowercased and combined directory name
#        DIR_NAME="${MODEL_NAME_LOWER}_${DATASET_NAME_LOWER}"
#
#        # Ensure directory exists
#        mkdir -p "./Networks/${DIR_NAME}"
#        mkdir -p "./Data/${DIR_NAME}"
#        mkdir -p "./Final_Data/${DIR_NAME}"
#
#        # Update MODEL_NAME, DATASET_NAME, and VALIDATION_ACCURACY_THRESHOLD in Python files
#        sed -i "" "s/^MODEL_NAME = .*/MODEL_NAME = '${MODEL_NAME}'/" General_Training.py
#        sed -i "" "s/^DATASET_NAME = .*/DATASET_NAME = '${DATASET_NAME}'/" General_Training.py
#        sed -i "" "s/^VALIDATION_ACCURACY_THRESHOLD = .*/VALIDATION_ACCURACY_THRESHOLD = ${THRESHOLD}/" General_Training.py
#
#        # The same update process should be repeated for other Python files as in your original script
#
#        # Run Python scripts with updated variables
#        python General_Training.py
#        python Generating_diffeo_images_General.py $MODEL_NAME $DATASET_NAME
#        python General_post_hoc.py $MODEL_NAME $DATASET_NAME
#
#        # Move and rename files according to the new variables
#        mv "./Networks/${MODEL_NAME_LOWER}_${DATASET_NAME_LOWER}.pth" "./Networks/${DIR_NAME}/save_${MODEL_NAME}_${DATASET_NAME}.pth"
#        mv "./Data/${MODEL_NAME_LOWER}_${DATASET_NAME_LOWER}.pth" "./Data/${DIR_NAME}/data_${MODEL_NAME}_${DATASET_NAME}.pth"
#        mv "./Final_Data/${MODEL_NAME_LOWER}_${DATASET_NAME_LOWER}/results.pth" "./Final_Data/${DIR_NAME}/results_${MODEL_NAME}_${DATASET_NAME}.pth"
#        mv "./Final_Data/${MODEL_NAME_LOWER}_${DATASET_NAME_LOWER}/eval_method.pth" "./Final_Data/${DIR_NAME}/eval_method_${MODEL_NAME}_${DATASET_NAME}.pth"
#
#    done
#done
