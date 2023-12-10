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
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── Dockerfile
    │
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
        └── shiny  <- Scripts to create exploratory and results oriented visualizations
           └── 'shiny'.py
    
Description of Steps 
------------

- Preparation of  a folder structure, using the cookiecutter Data Science template

- Downloaded of raw data [see below]

- Create upstream and downstream functions

- Generation of tables which can be directly imported to Latex 

- Connect Overleaf to Github (all Overleaf Files can be directly accessed [here](https://www.overleaf.com/read/yqkhbqjwvtbs#b7cd7c))

- Use R Shiny and Plotly to create interactive graphs in order to do robustness checks (data frequency, data features, date range, visualization) [here] (https://baumender11.shinyapps.io/Alpha/)

- Ensure reproducibility by adding Dockerfile 

- Analysis and interpretation of findings

- Compare results with current research

Data 
------------

Macroeconomic Attention Indices (MAI) data are obtained from [charlesmartineau/mai_rfs](https://github.com/charlesmartineau/mai_rfs) Github repository.

MEF data are obtained from the [powder197/Goyal-and-Welch-2008-](https://github.com/powder197/Goyal-and-Welch-2008-/tree/master) Github repository.

S&P500 proces are obtained from [Yahoo Finance](https://finance.yahoo.com).
  
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
