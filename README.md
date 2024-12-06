
# Final Project - CSI5170
### Matthew Couch and David Georgi

The following is the step needed to execute the code in this project. 
## Installation

Verify the following librarires are installed using the following commands:
```bash
  pip install numpy
  pip install pandas
  pip install sklearn
  pip install tensorflow
```
    
## Deployment

In order to test the project,  unzip the data.zip file and make sure the directory is called "data". 

You will need the activities.txt file and blood_glucose.csv file from this zipped folder to be in the "data" directory. The 30min average file will be created again later.

The python files need to be executed in the following order:

1. handle_missing_values.py

    i. This will create 'blood_glucose_30min_avg_keep_all_features.csv'

2. evaluate_preprocessing.py 

3. evaluate_original.py 

4. model_comparison.py 

    i. This will take a few hours to execute as we are tuning CNN models on a large dataset.

