# Mastermind-Elite
## 0. Install the requirements
Before run and execute the code, we need to install requirements.\
`pip install -r requirements.txt`

## 1.  Build the model
_classify.py_ is the script for creating the model start from do transformation 
of the dataset until build machine learning model and save the model.\
example:\
`python classify.py --imabance "oversampling" --model_name "random_forest" 
--max_depths "[20, 30, 40, 50]" --n_estimators "[100, 200, 300]"`
\
a. imbalance: sampling method for unbalance dataset, can be "oversampling" or
"undersampling"\
b. model_name: model name\
c. max_depths: number of max depth for hyperparameter tuning\
d. n_estimators: number of n_estimators for hyperparameter tuning

the model will be saved under model folder after you execute the script.\
the accuracy of the model will be saved and printed out in the log files under
log folder after you executed the model.

## 2. Load the model
_classifier.py_ is class for load the saved model and to be used in the main 
function

## 3. Evaluate the model
_main.py_ is the main function to evaluate the model. the output of the script
is confusion matrix and feature important of the model.\
`python main.py`