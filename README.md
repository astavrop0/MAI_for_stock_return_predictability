MAI_for_stock_return_predictability
==============================

We conduct forecasts for equity risk premia utilizing linear and neural network models trained on sets of macroeconomic factors and macroeconomic attention indices. The macroeconomic factors set comprises the 14 features recommended in previous works by Goyal and Welch (2008). The set of macro attention indices includes the eight features constructed by Fisher et al. (2022). Equity risk premia represent the annualized excess return of the one-month S&P 500 index over the prevailing risk-free rate, approximated by the yield on short-term Treasury Bills. 

Our analysis is based on datasets published in other Github repository and Yahoo Finance (see below), with focus on the period between 1985 and 2018, given the availability of the combined data.

Our results deviate from previous research, suggesting the need for further investigation into the datasets.

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
        │   │
        │   └── NN_model.py
        │
        └── shiny  <- Code and data to create R shiny app for visualizations
            │
            ├── shiny_data
            │
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

Macroeconomic Attention Indices (MAI) data are obtained from [charlesmartineau/mai_rfs](https://github.com/charlesmartineau/mai_rfs) Github repository.

MEF data are obtained from the [powder197/Goyal-and-Welch-2008-](https://github.com/powder197/Goyal-and-Welch-2008-/tree/master) Github repository.

S&P500 proces are obtained from [Yahoo Finance](https://finance.yahoo.com).

Installation
------------

To install the required packages for this project, follow these steps:

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

2. Navigate to the cloned repository's directory:

3. Build the Docker image. Replace `your-image-name` with a name of your choice for the Docker image:
```
docker build -t your-image-name .
```

### Running the Docker Container

Once the image is built, you can run it as a container. 

- To run the container, execute the following command:
docker run your-image-name

Resources
------------

Eugene F. Fama, Kenneth R. French, Dividend yields and expected stock returns, Journal of Financial Economics, Volume 22, Issue 1, 1988, Pages 3-25, ISSN 0304-405X (document can be directly accessed [here](https://www.sciencedirect.com/science/article/pii/0304405X88900207))

Goyal, Amit, and Ivo Welch. “Predicting the Equity Premium with Dividend Ratios.” Management Science, vol. 49, no. 5, 2003, pp. 639–54. JSTOR (document can be directly accessed [here](https://www.jstor.org/stable/4133989))

Andrew W. Lo & Manish Singh (2023) Deep-learning models for forecasting financial risk premia and their interpretations, Quantitative Finance, 23:6, 917-929 (document can be directly accessed [here](https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2203844))

Shihao Gu, Bryan Kelly, Dacheng Xiu, Empirical Asset Pricing via Machine Learning, The Review of Financial Studies, Volume 33, Issue 5, May 2020, Pages 2223–2273 (document can be directly accessed [here](https://dachxiu.chicagobooth.edu/download/ML.pdf))

Andrei, Daniel and Hasler, Michael, Investor Attention and Stock Market Volatility (June 3, 2013). The Review of Financial Studies, 2015 (document can be directly accessed [here](https://www.epfl.ch/labs/cfi/wp-content/uploads/2018/08/WP757_A2.pdf))

Jussi Nikkinen, Mohammed Omran, Petri Sahlström, Janne Äijö, Global stock market reactions to scheduled U.S. macroeconomic news announcements, Global Finance Journal, Volume 17, Issue 1, 2006, Pages 92-104, ISSN 1044-0283 (document can be directly accessed [here](https://www.sciencedirect.com/science/article/pii/S104402830600024X))

Feng Ma, Xinjie Lu, Jia Liu, Dengshi Huang, Macroeconomic attention and stock market return predictability, Journal of International Financial Markets, Institutions and Money, Volume 79, 2022, 101603, ISSN 1042-4431 (document can be directly accessed [here](https://doi.org/10.1016/j.intfin.2022.101603))


Contributors
------------

Aaron Arauz Baumender (@Baumender11), Michael Geiser (@mikegeiser), Andreas Stavropoulos (@astavrop0), Zhiyi Tang (@sukotzy)

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
