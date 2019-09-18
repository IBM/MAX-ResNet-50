#!/usr/bin/env bash

# ===============================================
#   Setup any hyperparameters here
# ===============================================
NUM_CLASSES=2

# ===============================================
#   Exit codes
# ===============================================
SUCCESS_RETURN_CODE=0
TRAINING_FAILED_RETURN_CODE=1
POST_PROCESSING_FAILED=2
PACKAGING_FAILED_RETURN_CODE=3

# ---------------------------------------------------------------
# Perform model training tasks
# ---------------------------------------------------------------

mkdir -p ${RESULT_DIR}/model

echo "# ************************************************************"
echo "# Training model ..."
echo "# ************************************************************"

python image_classification.py --output=${NUM_CLASSES}

# capture return code
RETURN_CODE=$?
echo "Return code from task 1: ${RETURN_CODE}"
if [ $RETURN_CODE -gt 0 ]; then
  # the training script returned an error; exit with TRAINING_FAILED_RETURN_CODE
  echo "Error: Training run exited with status code $RETURN_CODE"
  exit $TRAINING_FAILED_RETURN_CODE
fi
