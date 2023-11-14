
# Multinomial Logistic Regression: Asymptotic Normality on Null Covariates in High-Dimensions

Here we include the Python files that can reproduce all the figures presented in both the main text and the supplementary material. 

## Description

Description of files and folder:

* **lib_fast_solve.py** : computes the matrix 
$\sum_{i\in[n]} \mathsf{V}_i$ appeared in Theorems 2.1 and 2.2.
* **scatter.py** : generates and saves *scatter plots* with different values of $(n, p, K, q)$. 
This file is used to produce Figure 1 and Figure S3. 

* **qqplot.py** : generates and saves *Q-Q plots*  and  *histograms* for different $(n, p, K)$ with $q=1$ using $\Omega_{jj}$ or using  $\hat \Omega_{jj}$ 
under three different
covariate distributions (Gaussian, Rademacher, SNP). 
This file is used to produce Figures 2, 3, S1, and S2. 

* **real_data_example.ipynb**: conduct real data analysis on a heart disease data. 

* Folder **figs**: contains all the generated figures from **scatter.py** and **qqplot.py**. 



## Detailed executing steps

Executing the following steps will generate all the figures used in the main text and the supplement. The figures are saved to the folder **figs**. 

1. Run the script **scatter.py**: this step will generate all the scatter plots (Figure 1 in the main text and Figure S3 in the supplement). 

    We need to specify the `setting` in the script **scatter.py**: 

   * `setting` = 1: $(n, p, q) = (2000, 600, 1)$
   * `setting` = 2: $(n, p, q) = (3500, 1000, 1)$
   * `setting` = 3: $(n, p, q) = (5000, 1500, 1)$
   * `setting` = 4: $(n, p, q) = (2000, 600, 2)$
   * `setting` = 5: $(n, p, q) = (2000, 600, 4)$

    This will generate five PDF files with meaningful file names. 
    For example, the name 'scatter_Gaussian_true_n2000_p600_q1.pdf'
means the *scatter* plot generated from the data with *Gaussian* distributed covariates and the *true $\Omega_{jj}$* under the setting $(n, p, q) = (2000, 600, 1)$. 

2. Run the script **qqplot.py**: this step will generate all the Q-Q plots and histograms of P-values (Figures 2, 3 in the main text and Figures S1 and S2 in the supplement). 

    We need to specify the values of (`dist`, `omega`, `K`) in the script **qqplot.py**. 

    * There are three options for `dist`:
    `dist` = 'Gaussian' means covariates are sampled from Gaussian distribution; `dist` = 'Rademacher' means covariates are sampled from Rademacher distribution; `dist` = 'SNP' means covariates are sampled from SNP distribution. 
    The detailed description of these distributions are given in Section 3 of the main text.
    * There are two options for `omega`:
    `omega` = 'true' means we use the true $\Omega_{jj}$; `omega` = 'est' means we use the estimated $\Omega_{jj}$ (namely, $\hat \Omega_{jj}$ in equation (2.5) of the main text). 
    * There are three options for `K`:
    `K` = 2 means we consider the setting with $(n, p, K) = (4000, 1000, 2)$; `K` = 3 means we consider the setting with $(n, p, K) = (5000, 1000, 3)$; `K` = 4 means we consider the setting with $(n, p, K) = (6000, 1000, 4)$.

    Executing the script **qqplot.py** with the above combinations of (`dist`, `omega`, `K`) will generate PDF files with meaningful file names. 
    For example, 
    the name 'qqplots_Gaussian_true_K2.pdf' means the *Q-Q plot* generated from the setting with Gaussian covariates using the true $\Omega_{jj}$ with $K=2$,
    and the name 'hist_Gaussian_est_K2.pdf' means the histogram of p-values generated from the setting with Gaussian covariates using the estimated $\Omega_{jj}$. 

## Hardware
All the simulations were run on a cluster of 50 CPU-cores (each is an Intel Xeon E5-2680 v4 @2.40GHz) equipped with a total of 200 GB of RAM. 
## Execution time
* We used Python 3.9.6 to run all the simulations. 

* The execution times of scripts **scatter.py** and **qqplot.py** vary and can range from 2 minutes to 90 minutes. The specific duration depends on the values of $(n, p, K, q)$. Generally, running scripts with larger values of $(n, p, K, q)$ takes longer time. For example, running **scatter.py** with $(n, p, q) = (2000, 600, 1)$ takes approximately 2 minutes, while executing **qqplot.py** with $(n, p, K) = (6000, 1000, 4)$ takes around 90 minutes.