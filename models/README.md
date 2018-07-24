This is the folder for every machine learning model trained for the project.
The structure of each subfolder is the following:

- data/: contains train and test data. It is updated by `build.py` scripts;
- model/: contains all the needed for loading the model during the execution. It is updated by `train.py` scripts;
- `build.py`: generate new data split from db and save into data/;
- `train.py`: train again the model, it loads train data from data/. The output in model/
- `test.py`: test and evaluate the model. It loads the model from model/ and test data from data/
- `make.py`: execute all the above scripts.
- other modules for the model code.

In `__init__.py` there are important definitions (e.g. I/O functions, file parser etc.) and parameters declarations (model hyperparameters) that you might be interested in.

REMEMBER TO EXECUTE THOSE SCRITPS FROM THE ROOT FOLDER, otherwise anything will work.