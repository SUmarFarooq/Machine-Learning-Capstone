# Machine Learning Engineer Nanodegree
## Capstone Proposal
Syed Umar Farooq
October 31, 2018

## Proposal

### Domain Background

The internet age has given rise to a new mechanism of financing for creators and entrepreneurs: crowdfunding. By directly appealing to consumers, crowdfunding campaigns, a successful crowfunding campaign can create enormous media attention, establish a customer base, and provide a large influx of capital. For instance, Oculus, a virtual reality company, launched a prominent campaign in 2012 for its headset and raised more than 2.5 million dollars. Three years later, the company was bought by Facebook for two billion dollars.

Hundreds of crowdfunding websites have been established, but an early entrant into the market, Kickstarter, is the largest crowdfunding platform in the world. More than fifteen million people have backed a Kickstarter project, with the total amount funded being approximately four billion. That's four times what Facebook paid for instagram. 

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement

Like all forms of investing, crowdfunding has risks. Ambitious projects sometimes do not meet their funding target, which means they must refund their backers in full. A failed project can damage the company's credibility and drive backers away from that platform or crowdfunding generally. Therefore, it is important for both backers and and creators alike to understand how to run a successful crowdfunding campaign.

For more entrepreneurial projects, like some consumer technology projects, recent legal changes have made finding a successful project to support as an early adopter more lucrative. In 2017, [Regulation A+](https://techcrunch.com/2015/06/19/new-rules-allow-early-adopters-to-become-early-investors/), a section of the JOBS Act, came into effect and allowed companies to have mini-IPOs. This process can cost up to 90% less than a full IPO and be completed in a few months rather than a few years, giving early backers of major projects the chance to become investors and giving projects access to individuals that can serve as shareholders and brand advocates.

Still, [63% of projects](https://www.statista.com/statistics/235405/kickstarter-project-funding-success-rate/) on Kickstarter fail to meet their funding goals, and [13% never receive a single pledge](https://www.kickstarter.com/help/stats). This rate is even higher for some types of projects. Technology projects,for instance, fail to meet their funding goals nearly 80% of the time. It is apparent that many project leaders are not able to successfully run a crowdfunding campaign, and many backers are not able to successfully back projects with a strong chance of being fully funded. **What factors, then, determine a project's likelihood of being successfully funded?**

 it is still unclear how to pick winners and losers among the more than 400,000 projects currently on Kickstarter. Some research has been done on the expressions used in [project descriptions](https://dl.acm.org/citation.cfm?id=2531656) while other research has focused on how the [gender](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2442954) of project leaders affects rates of successful adoption. Yet most research 

 It's getting more and more important because now early adopters have the ability to benefit financially (bc their early adoption can make or break a product, if a product goes big, new laws allow them to benefit)
From a launcher's perspective, it may be helpful in setting more reasonable goals. e.g., do you aim big for kickstarter goals or small? Seeing how it works would be really interesting 
So, what factors determine if a project does well?
There's really interesting academic discussions of how the language used or the launcher's gender affects success. While we do not have access to that data, we think that's also relevant

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs

To answer the above question we will explore a dataset of more than 200,000 Kickstarter campaigns. The data, hosted by [Kaggle](https://www.kaggle.com/kemical/kickstarter-projects), includes the project's mame, ID, main category (e.g. technology), subcategory (e.g. hardware). the number of backers, the amount of money raised, the projects goal, the project's country of origin, and if the project was successful in meeting its funding goal (also known as project status). The project status will be used as the dependent variable, with ongoing or canceled projects being removed from the data set. This will make project status a binary dependent variable.

To determine the value of project status, a combination of categorical and continuous variables will be used. Continuous variables will include project goal and the number of backer. If projects with a large goal are more successful, it could suggest those campaigns have the most resources. If not, a large goal could indicate a project overestimating its capacity to raise funds. Number of backers can indicate the popular support for a project, and might be associated with a larger chance of project success. 

Categorical variables will include subcategory and country of origin. Category will be restricted to a single value due to the sheer size of the data set and because there are dozens of potential subcategories. Restricting the category allows for a clearer analysis of the relevant factors. Because many high profile Kickstarter campaigns and acquisitions have been of technology campaigns, and because technology campaigns have the highest rate of failure, technology will be the category examined in this project. 

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement

I propose predicting the likelihood of a project's success using bagged and boosted random forests. This can be measured through an F-score or by comparing the accuracy on the training set to that of the cross validation set.
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model

The standard approach to problems with binary dependent variables is to use a logistic regression model. Therefore, the logistic regression model. This can be measured through an F-score or by comparing the accuracy on the training set to that of the cross validation set.
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics

Fortunately, both the benchmark model and the solution statement can be measured using standard evaluation metrics for supervised learning techniques. I intend to compare the F-scores and accuracies for the benchmark and solution models on training and cross validation sets.

Accuracy measures the percentage of data points our model predicts that are the same as the data's actual value. Its values can have a complicated meaning, though. If the data fits the training set very well but performs poorly on the testing set, that suggests the model is overfitting on the training data.

An F-score is a metric that measures how many of our true positives were correct. It takes into account both recall and precision. In our example, recall would be the percentage of funded projects which were correctly identified by our model, whereas precision would be the percentage of projects that our project predicted were fully funded which were actually fully funded. F-score takes in a parameter referred to as beta. This parameter specifies the tradeoff between recall and precision. This problem in this proposal is broad enough to care about both recall and precision, but the main interest is in avoiding projects that are failures, so we will set beta to try and maximize precision.

_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design

#### Exploratory Data Analysis
Before formally analyzing the data, I'd like to visualize much of the data. What do rates of success look like across different countries? Were early kickstarter campaigns more likely to fail? What percentage of projects succeed across various technology subcategories? Using bar charts I can visualize the number of successes and failures and hope to compare those numbers across various segments.

#### Data Preprocessing
The data needs to be cleaned up a bit before the work can begin. The current dataset contains data that is not needed for regression, such as the starting and ending dates for each crowdfunding campaign. I do not plan to use them because of the difficulty of incorporating dates into the models I have outlined. Other variables, such as currency, will be dropped due to the presence of a similar enough proxy (country).

I also need to convert my categorical variables into a series of binary explanatory variables through one-hot encoding.

Finally, I'll have to normalize my continuous variables. number of backers for Kickstarter projects can range from zero to the thousands, and the goals can range from five hundred dollars to hundreds of thousands of dollars. Normalizing this data through a log-transformation should reduce how skewed the data is.

#### Data Analysis 

The fun part! Here, I will use both the benchmark and solution model and run them on sets of training data of various sizes (e.g. 10% and 20% of the total dataset.) Then, I will attempt to tune both models using the training and cross validation sets to find better parameters(e.g., I'll see if changing the bagging or boosting parameters will increase the accuracy scores without leading to overfitting). 

_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?