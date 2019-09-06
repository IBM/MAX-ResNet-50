## How to Train Image Classification Model Using Your Own Data

- [Collect Data for Training](#collect-data-for-training)
- [Train the Model](#train-the-model)
- [Rebuild the Model-Serving Microservice](#rebuild-the-model-serving-microservice)


## Collect Data for Training

Collect RGB images encoded as `jpeg` or `png` containing objects to be classified. Make sure
the training images have large variations in angle, resolution, lighting and background so that they generalize 
well with the test data. Use a reasonably large number if images per class to provide better results.

Create folders equal to number of classes. Folder names should be same as the class names. For example if we need to
classify dog and cat images, create two folders with names `dog` and `cat` and place `dog` and `cat` images in the 
respective folders.

## Train the Model

- [Install Local Prerequisites](#install-local-prerequisites)
- [Run the Setup Script](#run-the-setup-script)
- [Train the Model Using Watson Machine Learning](#train-the-model-using-watson-machine-learning)

In this document `$MODEL_REPO_HOME_DIR` refers to the cloned MAX model repository directory, e.g.
`/users/hi_there/MAX-Resnet50`. 

### Install Local Prerequisites

Open a terminal window, change dir into `$MODEL_REPO_HOME_DIR/training` and install the Python prerequisites. (Model training requires Python 3.6 or above.)

   ```
   $ cd training/

   $ pip install -r requirements.txt
    ... 
   ```
   
To test the model training process, use data from `/sample_training_data` and skip data preparation step.

### Run the Setup Script

#### Purpose

In order to run the model training script two sets of environment variables need to be defined:

##### 1. Watson Machine Learning

- ML_APIKEY
- ML_ENV
- ML_INSTANCE

##### 2. Cloud Object Storage

- AWS_ACCESS_KEY
- AWS_SECRET_ACCESS_KEY

The wml_setup.py script (among other things) ensures that these variables are properly defined 
and YAML file is properly configured. 

Input training data bucket, result bucket, local directory from where data will be uploaded and GPU 
configuration are the details that will be updated in YAML file.

The main menu options vary depending on which environment variables are set when wml_setup.py is run.

#### Steps

1. Locate the training configuration file. It is named `...-training-config.yaml`.

   ```

   $ ls *.yaml
     <...-training-config.yaml> 
   ```

2. Configure your environment for model training.

   ```
    $ python wml_setup.py <...-training-config.yaml> 
     ...
   ```
   
3. Once setup is completed, define the displayed environment variables.

   MacOS/Linux example:

   ```
   $ export ML_APIKEY=...
   $ export ML_INSTANCE=...
   $ export ML_ENV=...
   $ export AWS_ACCESS_KEY_ID=...
   $ export AWS_SECRET_ACCESS_KEY=...
   ```
   
   Also, note the YAML configuration.
   
   ```
       ------------------------------------------------------------------------------
       NEW YAML CONFIGURATION VALUES
       ------------------------------------------------------------------------------
       input_bucket  : resnet50-input
       local directory  : .../inp_obj
       result bucket  : resnet50-output
       compute  : k80
   ```

### Train the Model Using Watson Machine Learning

#### Purpose

- To initiate training in Watson Machine Learning.
- To download model and log files.
- Move the downloaded files to the parent directory so they are included in the Docker image


#### Commands

1. Verify that the training preparation steps complete successfully. Replace `<model-name.yaml>` with your configuration file.

   ```
    $ python wml_train.py <...-training-config.yaml> prepare
     ...
     # --------------------------------------------------------
     # Checking environment variables ...
     # --------------------------------------------------------
     ...
   ```

   If prepartion completed successfully:

    - Training data is present in the Cloud Object Storage bucket that WML will access during model training.
    - Model training code is packaged `<model-name>-model-building-code.zip`

2. Start model training.

   ```
   $ python wml_train.py <...-training-config.yaml> package
    ...
    # --------------------------------------------------------
    # Starting model training ...
    # --------------------------------------------------------
    Training configuration summary:
    Training run name     : train-max-...
    Training data bucket  : ...
    Results bucket        : ...
    Model-building archive: max-...-model-building-code.zip
    Model training was started. Training id: model-...
    ...
   ```
   
   > Note the `Training id` displayed.

3. Monitor training progress

   ```
   ...
   Training status is updated every 15 seconds - (p)ending (r)unning (e)rror (c)ompleted: 
   ppppprrrrrrr...
   ```

   After training has completed the training log file `training-log.txt` is downloaded along with the trained model artifacts.

   ```
   ...
   # --------------------------------------------------------
   # Downloading training log file "training-log.txt" ...
   # --------------------------------------------------------
   Downloading "training-.../training-log.txt" from bucket "..." to "training_output/training-log.txt"
   ..
   # --------------------------------------------------------
   # Downloading trained model archive "model_training_output.tar.gz" ...
   # --------------------------------------------------------
   Downloading "training-.../model_training_output.tar.gz" from bucket "..." to "training_output/model_training_output.tar.gz"
   ....................................................................................
   ```

   > If training was terminated early due to an error only the log file is downloaded. Inspect it to identify the problem.

   ```
   $ ls training_output/
     model_training_output.tar.gz
     trained_model/
     training-log.txt 
   ```
 
   To **restart** monitoring, `python wml_train.py <...-training-config.yaml> package <training id>`.
  
   To **cancel** the training run, press `ctrl+C` twice.

4. Return to the parent directory

## Rebuild the Model-Serving Microservice

Once the training run is complete, there should be a `frozen_inference_graph.pb` and `label_map.pbtxt` files in 
$MODEL_REPO_HOME_DIR/custom_assets folder.

The model-serving microservice out of the box serves the pre-trained model which was trained on COCO dataset. 
To serve the model trained model on your dataset you have to rebuild the Docker image:

1. Rebuild the Docker image

   ```
   $ docker build -t <max-model-name> --build-arg use_pre_trained_model=false . 
    ...
   ```
   
   > If the optional parameter `use_pre_trained_model` is set to `true` or if the parameter is not defined the Docker image will be configured to serve the pre-trained model.
   
 Once the Docker image build completes you can start the microservice as usual:
 
 ```
 $ docker run -it -p 5000:5000 <max-model-name>
 ```
