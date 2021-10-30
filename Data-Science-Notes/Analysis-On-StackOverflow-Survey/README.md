# Data Science Blog

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Components](#project-components)
3. [File Description](#file-description)
4. [Installation](#installation)
5. [Discussions](#discussions)

## Project Overview
Guided by the __CRISP-DM__ process, this project includes an analysis of the data from [Stack Overflow Annual Developer Survey](https://insights.stackoverflow.com/survey) and a Data Science Blog on [Medium](https://medium.com/@yukirinssgh/what-fe4318c22795). Specially, we aim to find out answers to the following three questions:

- What proportions of developers work as data scientists?
- What programming languages do data scientists use mostly?
- What and Why should data scientists learn besides Python and SQL?

## Project Components
There are two components in the project.

### 1. Exploratory Data Analysis 
In order to answer the three questions above, the whole process of an exploratory data analysis can be found in the notebook.
- Load data from csv, extract interested columns and rows, drop missing rows.
- Create dummies for devTypes, compute ratio of each devType and visualize.
- Create dummies for languages, compute language popularity and visualize.

### 2. Data Science Blog
I also wrote a data science blog as a output of the data analysis above. The blog can be found on [__Medium__](https://medium.com/@yukirinssgh/what-fe4318c22795). 


## Installation
### Devendencies :
   - [python (>=3.8)](https://www.python.org/downloads/)   
   - [pandas](https://pandas.pydata.org/)  
   - [numpy](https://numpy.org/)  
   - [matplotlib](https://matplotlib.org/stable/users/installing.html)

## File Description
```sh
- README.md: read me file
- \img
	- images used \in the blog.
- AnalysisOnStackOverflow.ipynb: Exploratory Data Analysis 
```

## Discussions

(The whole article can be found on [__Medium__](https://medium.com/@yukirinssgh/what-fe4318c22795).)

<div align=center>
<img src="https://github.com/BlickWinkel17/Data-Science-Blog/blob/master/img/ratios_DevTypes.png" height="300"> 
<div>
<div align=center>
<img src="https://github.com/BlickWinkel17/Data-Science-Blog/blob/master/img/languages_worked_with.png" height="300">
<div>

we took a look at skills really needed by data scientists in industry according to Stack Overflow 2021 survey data.
- We first looked at the overview of data industry in 2021. It showed that data science seemed to become less popular, but is still significant, especially considering the impact of pandemic.
- We then looked at the most preferred programming languages are preferred by data scientists. SQL and Python are powerful but more data scientists are picking up front-end languages.
- Finally, we discussed that why and what data analysts should learn besides Python and SQL. Python may serve as the best stepping stone to data science industry, but keep learning more skills keep you competitive as a data scientist.
