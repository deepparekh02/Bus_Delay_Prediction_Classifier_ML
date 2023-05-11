DESCRIPTION
This repo makes a predictive model on bus delays in Toronto using features like routes, time, direction.
Possible uses can be real-time information update for internal transit purposes and to update transit users immediately about estimated delay times.


FILES

- main.ipynb 
Contains all code ranging from data importing to model performance evaluation. Ideally would be divided into 3 python files: data_setup, data_processing, modelling; 
Done this waybecause of limited computational capacity and time contraints.

- ttc-bus-delay-data-2020.xlsx, ttc-bus-delay-data-2021.xlsx, ttc-bus-delay-data-2022.xlsx
Raw datasets used to make model.

- ttc-bus-delay-data-readme.xlsx
Containts information about raw datasets.

- model.pkl
Saved Random Forest classifier model to prevent the need to train again.

- classification_report.png, confusion_matrix.png
Model performance evaluation outputs.