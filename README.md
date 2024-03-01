> To create virual environment: python: python -m venv ```name of the environment (env)```
To activate the environment: ```name(env)/Scripts/activate```


For Conda environment: ```conda create -n env python==3.7 -y```
For conda environment activation: ```activate env``` or ```conda activate env```


The next: The dependencies are found in the file ```requirements.txt```
```pip install -r requirements.txt```

Read the dataset using pd.read_csv("data/spamhamdata.csv")

In Train dataset, we use TfIdfvectorizer.fit_transform, but in test, we use transform to avoid overfitting,
Overfitting is the ability of a model to be able to generalize well on train dataset and not well on test dataset
Underfittting, the model does not generalize well on the train dataset

y = wixi + b

> Inside ```app.py```, we have streamlit app, to run this app for user consumption, we should rubn the following command. ```streamlit run app.py```

