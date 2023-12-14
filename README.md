MAI_for_stock_return_predictability
==============================

We conduct forecasts for equity risk premia utilizing linear and neural network models trained on sets of macroeconomic factors (MEF) and macroeconomic attention indices (MAI). The macroeconomic factors set comprises the 14 features recommended in previous works by Goyal and Welch (2008). The set of macro attention indices includes the eight features constructed by Fisher et al. (2022). Equity risk premia represent the annualized excess return of the one-month S&P500 index over the prevailing risk-free rate, approximated by the yield on short-term Treasury Bills. 

Our analysis is based on datasets published in other Github repositories and Yahoo Finance (see below), with focus on the period between 1985 and 2018, given the availability of the combined data. We provide an [interactive app](https://baumender11.shinyapps.io/Alpha/) for the user to undertake robustness checks of the data.

Our results of our prediction models deviate from previous research, suggesting the need for further investigation into the datasets.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project
    ├── data
    │   ├── interim        <- Intermediate data (filled missing values, droped useless features)
    │   ├── processed      <- The final, canonical data sets for modeling (further transformations)
    │   └── raw            <- The original, immutable data dump
    │
    ├── notebooks              <- Jupyter notebooks
    │
    ├── reports                <- Generated analysis as PDF and LaTeX report and beamer presentation
    │   └── figures            <- Generated graphics and figures to be used in reporting as .png
    │
    ├── requirements.txt       <- The requirements file for reproducing the analysis environment
    │
    ├── Dockerfile             <- Defines the Docker container configuration to run the analysis
    │
    └── src                    <- Source code for use in this project
        ├── __init__.py        <- Makes src a Python module
        │
        ├── data               <- Scripts to turn raw data into features for modeling
        │   └── make_dataset.py
        │
        ├── models             <- Scripts to train evaluate models             
        │   ├── linear_model.py
        │   └── NN_model.py
        │
        └── shiny              <- Code and data to create R shiny app for visualizations
            ├── shiny_data
            └── stock_return_prediction_app.R
    
Description of Steps 
------------

- Preparing of a folder structure, using the cookiecutter Data Science template

- Downloading raw data [see below]

- Creating upstream and downstream functions

- Generating tables which can be directly imported to Latex 

- Connecting Overleaf to Github (all Overleaf Files can be directly accessed [here](https://www.overleaf.com/read/yqkhbqjwvtbs#b7cd7c))

- Creating an interactive app in Shiny For R to perform robustness checks - data frequency, data features, date range, visualization - (app can be directly accessed [here](https://baumender11.shinyapps.io/Alpha/))

- Ensuring reproducibility by adding Dockerfile 

- Analysing and interpreting findings

- Comparing results with current research

Data 
------------

Macroeconomic Factors (MEF) data is obtained from the [powder197/Goyal-and-Welch-2008-](https://github.com/powder197/Goyal-and-Welch-2008-/tree/master) Github repository.

Macro Attention Indices (MAI) data is obtained from [charlesmartineau/mai_rfs](https://github.com/charlesmartineau/mai_rfs) Github repository.

S&P500 data is obtained from [Yahoo Finance](https://finance.yahoo.com).

Installation
------------

To install the required packages nd replicate this project, follow these steps:

1. Clone the repository to your local machine:
```
  git clone https://github.com/astavrop0/MAI_for_stock_return_predictability.git
```
2. Navigate to the cloned repository directory:
```
  cd MAI_for_stock_return_predictability
```
3. Install the required packages using pip:
```
  pip install -r requirements.txt
```
  This will install all the Python packages listed in `requirements.txt`.

Running with Docker
------------

This project can be easily set up and run using Docker. Follow the steps below to build and run the project inside a Docker container.

### Prerequisites
1. Ensure you have [Docker installed](https://docs.docker.com/get-docker/) on your machine.

2. Navigate to the cloned repository's directory.

3. Build the Docker image. Replace `your-image-name` with a name of your choice for the Docker image:
```
docker build -t your-image-name .
```

### Running the Docker Container

Once the image is built, you can run it as a container. To do it, execute the following command:
```
docker run your-image-name
```

Resources
------------

[[1]](https://www.sciencedirect.com/science/article/pii/0304405X88900207) Eugene F. Fama, Kenneth R. French, Dividend yields and expected stock returns, Journal of Financial Economics, Volume 22, Issue 1, 1988, Pages 3-25, ISSN 0304-405X

[[2]](https://www.jstor.org/stable/4133989) Goyal, Amit, and Ivo Welch. “Predicting the Equity Premium with Dividend Ratios.” Management Science, vol. 49, no. 5, 2003, pp. 639–54. JSTOR

[[3]](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2203844) Andrew W. Lo & Manish Singh (2023) Deep-learning models for forecasting financial risk premia and their interpretations, Quantitative Finance, 23:6, 917-929

[[4]](https://dachxiu.chicagobooth.edu/download/ML.pdf) Shihao Gu, Bryan Kelly, Dacheng Xiu, Empirical Asset Pricing via Machine Learning, The Review of Financial Studies, Volume 33, Issue 5, May 2020, Pages 2223–2273

[[5]](https://www.epfl.ch/labs/cfi/wp-content/uploads/2018/08/WP757_A2.pdf) Andrei, Daniel and Hasler, Michael, Investor Attention and Stock Market Volatility (June 3, 2013). The Review of Financial Studies, 2015

[[6]](https://www.sciencedirect.com/science/article/pii/S104402830600024X) Jussi Nikkinen, Mohammed Omran, Petri Sahlström, Janne Äijö, Global stock market reactions to scheduled U.S. macroeconomic news announcements, Global Finance Journal, Volume 17, Issue 1, 2006, Pages 92-104, ISSN 1044-0283

[[7]](https://doi.org/10.1016/j.intfin.2022.101603) Feng Ma, Xinjie Lu, Jia Liu, Dengshi Huang, Macroeconomic attention and stock market return predictability, Journal of International Financial Markets, Institutions and Money, Volume 79, 2022, 101603, ISSN 1042-4431


Contributors
------------

Aaron Arauz Baumender (@Baumender11), Michael Geiser (@mikegeiser), Andreas Stavropoulos (@astavrop0), Zhiyi Tang (@sukotzy)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
