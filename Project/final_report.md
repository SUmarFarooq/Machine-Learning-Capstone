# Machine Learning Engineer Nanodegree
## Capstone Project
S. Umar Farooq
November 21st, 2018

## I. Definition

### Project Overview

Kickstarter is the largest crowdfunding platform in the world. More than fifteen million people have backed a Kickstarter project, with more than four billion dollars of funding being pledged since the site launched almost ten years ago. Some of the most successful projects have gained not just funding, but a large customer base and media coverage. Oculus, for instance, launched a campaign for its headset in 2012, raised 2.5 million dollars, and was acquired by Facebook for two billion dollars just three years later. 

### Problem Statement

There are several benefits to being an early backer of a major Kickstarter project. In addition to receiving perks, backers have the potential to become early investors in the product, which can have an enormous upside if the project is a technology project. Yet it is incredibly difficult to tell which projects will be successful in meeting their fundraising goals. This paper attempts to determine a project's likelihood of being successfully funded and determine the optimal way to make such.

This is a binary classification problem, where the dependent variable is whether or not the project was successfully funded. Many prominent Kickstarter projects have launched companies or become major acquisitions by industry leaders. Because I am most interested in these projects, and many of them involve consumer technology, I will focus specifically on Technology projects. I believe the likelihood of a project being successfully funded will depend largely on the size of the fundraising goal, with projects with smaller goals being more likely to meet their funding goals. I also believe projects with a longer campaign will be more likely to succeed. I believe the category of project will also be important (hardware, software, apps, etc.) but I do not know which categories might be the most likely to succeed.

After cleaning and transforming my data, I will analyze it using a logistic regression as a benchmark model. I will then attempt to improve on this classification using bagged and boosted decision trees, as well as a boosted random forest. After this, I will extract the most important features and explore the relationships they have with the outcome variable.

### Metrics

Logistic regression, random forests, and ensemble methods all have accuracy and F-score as possible evaluation methods. Using the testing results of these metrics, I can accurately compare each model to see which has the highest level of predictive power.

Determining what one sets beta as when evaluating each model's F-beta score is tough. Changing beta weighs precision and recall differently. Either can be important depending on the target audience. If an investor is creating this model, they may prefer a model with a greater emphasis on precision, because an investor has a limited amount of money, and if they invest in projects they expect to succeed but don't actually succeed, it could be bad for their portfolio. For the average person, though, precision and recall might be weighted equally, since an average person does not have a strong preference between the two. Because my angle is somewhat more entrepreneurial, I will weigh precision more heavily and set beta to equal 0.5.

For determining which features are the most important, I intend to use the feature_importance metric available for the random forest classifier. I then intend to explore these features and their effect on the data.

## II. Analysis

### Data Exploration
My data set on available Kickstarter projects was obtained through Kaggle. In the full data set, there are 378,661 data points, though the number drops to 331,675 after the data set is limited to only projects with a successful or failed status (excluding projects which were canceled or are ongoing). That number decreasing even further once the data set is limited to technology projects, with the total number of observations dropping to 27,050.

The data set has both continuous and categorical features. Categorical features include subcategory (e.g. hardware, software, apps), currency, and country where the campaign was launched. Continuous features include the fundraising target (goal), the amount of money pledged, the date the campaign was launched, the campaign deadline,and the number of backers. 

Below are summary statistics and the first few entries for the data after filtering out non-technology projects and projects whose outcome was neither successful nor failed.. 

#### Figure 1: Summary statistics for working data

|       | **backers**  | **usd_pledged_real** | **usd_goal_real** |
|-------|--------------|----------------------|-------------------|
| count | 27050.00000  | 2.705000e+04         | 2.705000e+04      |
| mean  | 186.14695    | 2.397673e+04         | 9.807654e+04      |
| std   | 1228.11862   | 1.359790e+05         | 1.591295e+06      |
| min   | 0.00000      | 0.000000e+00         | 7.500000e-01      |
| 25%   | 1.00000      | 1.100000e+01         | 5.009007e+03      |
| 50%   | 7.00000      | 3.600000e+02         | 1.800000e+04      |
| 75%   | 54.00000     | 5.515750e+03         | 5.000000e+04      |
| max   | 105857.00000 | 6.225355e+06         | 1.101698e+08      |

#### Figure 2: Assorted data points for working data
|     | **category** | **deadline** | **launched**        | **state** | **backers** | **country** | **usd\_pledged\_real** | **usd\_goal\_real** |
|-----|--------------|--------------|---------------------|-----------|-------------|-------------|------------------------|---------------------|
| 65  | Hardware     | 2015-07-03   | 2015-06-03 05:52:43 | failed    | 0           | CA          | 0.00                   | 39739.31            |
| 67  | Software     | 2017-07-02   | 2017-06-02 12:20:21 | failed    | 0           | GB          | 0.00                   | 2579.35             |
| 71  | Web          | 2016-08-23   | 2016-07-24 13:18:36 | failed    | 3           | US          | 141.00                 | 100000.00           |
| 98  | Gadgets      | 2015-03-07   | 2015-02-05 16:57:21 | failed    | 3           | CA          | 2.36                   | 19632.48            |
| 112 | Gadgets      | 2017-06-14   | 2017-05-10 16:00:18 | failed    | 6           | US          | 74.00                  | 500.00              |

A few things are noteworthy here. First, the majority of projects have very few (less than 10) backers and receive less than $400 of funding. Second, the range of the continuous variables is immense. Backers range from 0 to 105,857, dollars pledged ranges from zero to more than six million, and goals range from 75 cents to 100 million dollars. Some work will be needed to deal with these outliers. Lastly, though it is not apparent from these figures, the data set does not require extensive cleaning. There do not appear to be instances of or incomplete data for either the categorical or continuous variables. 

### Exploratory Visualization
# TODO
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques

As mentioned before, I will use bagged and boosted decision trees, boosted and unboosted random forests, and logistic regression to determine which algorithm has the highest predictive power in determining the relevant features in predicting campaign success.

The reasons for choosing these algorithms is relatively straightforward. Using ensemble methods allows us to reduce the likelihood of overfitting and reduce the variance in the model by training several weak learners. The reason for using both random forests and bagged trees is because random forests only use a subset of the features to train the weak learners. I was unsure if only training on a subset of features might leave out important data or if it would reduce unnecessary noise, so I used both versions.

Each model has a range of hyperparameters that can be picked. I will use sklearn's GridSearchCV tool to pick most of the optimal hyperparameters for each model. I will prioritize hyperparameters which I think will have a noticeable impact on how the model performs (e.g., the minimum number/percentage of samples that need to be in the leaf of a decision tree or random forest.) I will also exclude parameters which I do not think make sense to evaluate using GridSearchCV. For instance, AdaBoost takes in a parameter for the number of estimators, and it is expected that a larger number of estimators will consistently yield better estimates.

### Benchmark

The benchmark model I will use to compare against my random forest and ensemble methods is a logistic regression. This is because logistic regression is considered to be a good starting point for most binary classification problems. It is quick to train, has high predictive capability, and is relatively robust for several use cases. As mentioned above, I will use GridSearchCV to also optimize the hyperparameters for the logistic regression model so I can fairly compare it with the models I am testing out.

As for setting a default F-score or accuracy measurement as the standard, I do not have a prior distribution or probability I can use to determine how much variance our model should account for. My hope is that the models can achieve higher than 50% accuracy, but this is not sufficient reason to set this as our benchmark.

## III. Methodology

### Data Preprocessing

I began by dropping unnecessary variables, such as the IDs, names, and currency of funds raised of each kickstarter project. I also dropped variables containing each project's goal or amount pledged and opted to instead use the amount of money in real US dollars for consistency between projects in different countries.

Next, I used Python's datetime library to generate a feature for the length of each fundraising campaign using the date launched and deadline for each campaign. Summary statistics for that variable, named "campaign_length" are below.

|       | **campaign\_length** |
|-------|----------------------|
| count | 27050.000000         |
| mean  | 35.264288            |
| std   | 11.771582            |
| min   | 1.000000             |
| 25%   | 30.000000            |
| 50%   | 30.000000            |
| 75%   | 40.000000            |
| max   | 92.000               |

The data had three continuous variables that all seemed indicative of success: the campaign's fundraising goal, the amount of funds pledged, and the number of backers. I expected the funds pledged and the number of backers to be positively correlated with success, but felt uncertain about leaving both the funds pledged and the goal in as features. Using both of them would allow an individual to predict every data point. Yet it also seemed excessive to drop either variable; each contained potentially valuable explanatory data. To get around this, I created a variable for the dollars raised per backer, by dividing the amount pledged by the number of backers. This allowed me to retain the data contained within the features without risking exposing too much information. I then removed both of the original variables, leaving me with features for the average pledge per backer and the goal of each fundraising campaign. 

An observant reader might notice that creating the above variable could be problematic, as there were several instances where projects had no backers. On the advice of mentors, I opted to drop instances where campaigns had zero backers.

The next major hurdle was to deal with outliers in my data. This was most problematic in the feature for each campaign's fundraising goal. With some campaigns having goals as high as 100 million dollars, the variation in this feature could dramatically skew the distribution of the data. I saw that approximately a little under 10% of my observations had goals over 200,000 dollars, but more than of of those observations had less than ten backers, suggesting that the campaigns were outlandish in their goals. This is substantiated when examining [some](https://www.kickstarter.com/projects/2099347793/hydroponics-skyscraperun-gratte-ciel-hydroponiquee?ref=discovery&term=hydroponics) of the projects with the largest goals on Kickstarter. Since my capstone is focused on serious projects, campaigns with very outlandish goals arguably fall outside of the scope of my research, but they can also obfuscate the patterns in the data I am interested in. I removed the roughly five percent of observations with goals above 200,000 dollars, and removed any observations with goals above 100,000 dollars with less than ten backers.

This step addressed a large amount of the variance in one of my features, but not the rest of them. To address the variance more generally, I then log transformed and scaled the data.

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?