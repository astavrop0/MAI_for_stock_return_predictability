MAI_for_stock_return_predictability
==============================

In this project we are examining whether the macroeconomic attention indices proposed by Fischer et al. (2022) have the ability to predict stock returns based on some statistical and machine learning algorithms. To evaluate that we compare their predictability efficiency with that of other popular macroeconomic variables.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to turn raw data into features for modeling
        │   └── make_dataset.py
        │
        ├── models         <- Scripts to train models and make predictions
        │   │                 
        │   ├── linear_model.py
        │   └── NN_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py
    
  
Resources
------------

Adlai Fisher, Charles Martineau, Jinfei Sheng, Macroeconomic Attention and Announcement Risk Premia, The Review of Financial Studies, Volume 35, Issue 11, November 2022, Pages 5057–5093, https://doi.org/10.1093/rfs/hhac011

Feng Ma, Xinjie Lu, Jia Liu, Dengshi Huang,
Macroeconomic attention and stock market return predictability,
Journal of International Financial Markets, Institutions and Money, Volume 79, 2022, 101603, ISSN 1042-4431, https://doi.org/10.1016/j.intfin.2022.101603

Contributors
------------

Aaron Arauz Baumender (@Baumender11), Michael Geiser (@mikegeiser), Andreas Stavropoulos (@astavrop0), Zhiyi Tang (@sukotzy)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
