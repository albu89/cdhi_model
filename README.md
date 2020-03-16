cdhi_model
==============================

To meet the objective of building a android deployable speech classifier that can distinguish speaking, from singing and silence a convolutional net on the raw audio signal was trained inspired by the work of **[Wei Dai. et al.](https://drive.google.com/file/d/1G040rNPvGnjRTqXe1Tc6GFooibk2u7v7/view?usp=sharing)**. More information on the model and the results can be found **[here.](https://github.com/albu89/cdhi_model/blob/master/docs/CDHI%20Challenge%20V1.0.pdf)**


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data from eg. data interpretation.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Project documentation
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks of data exploration
    │
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── a_create_deata.py    <- Scripts to prepare data for modelling
    │   │   
    │   │
    │   ├── b_train_model.py     <- Scripts to run experiment and save results
    │   │   └── build_features.py
    │   │


### Building

### Prerequisites

#### Install virtualenvwrapper (Windows)

Run `pip install virtualenvwrapper-win` in the command line tool.

#### Install virtualenvwrapper (Mac)
First install virtualenvwrapper with:
```
pip install virtualenvwrapper
```
Next, locate your virtualenvwrapper.sh file with:
```
which virtualenvwrapper.sh
```
Note the returned path and use it in the next step:
Open ~/.bash_profile (~ is the default directory in your terminal) with your favorite editor. 
If the file does not exist, create it. Append the following to the existing content:
```
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /path/to/your/virtualenvwrapper.sh
```
Save the changes and close the file.
Now, initialize the changes by:
```
source ~/.bash_profile
```

#### Create virtual environment
```
mkvirtualenv -p python364 env1
``` 
If this fails, try:
```
mkvirtualenv env1
``` 

#### Sidenote: Using virtualenvwrapper
You can use the following commands to list, activate and deactivate existing environments.
```
# Return a list of all existing virtual environments
lsvirtualenv

# Activate the selected environment
workon name_of_your_env 

# Deactivate an activated environment
deactivate
```
If you did not just create the new environment, activate the virtual environment you want to work in now by `workon name_of_your_env`.

#### Installing requirements

Install the dependencies listed in the requirements.txt file at the root folder by executing the following command line in the anaconda console:

```
pip install -r requirements.txt
```

If you find you need to install another package

```
pip freeze > requirements.txt
```

and commit the changes to version control.


### Creating a model

A step by step series of examples that tell you how to create a model.

+ Select the features in the `a_prepare_data_config.json` file that you want to have processed. An overview of all features can be found in the raw_data folder
+ Execute the crate data script

```
python a_create_data.py
```

+ Specify the model parameters in the `b_FF_model_config.json` file along with the processed dataset to be used in the data_set folder
+ Define the evaluation metrics you want to use in the `c_evaluation_config.json` file
+ Execute the run model script 

```
python b_run_model.py
```
+ Investigate the results in the model folder
