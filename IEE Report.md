 



Analysis of EEG Data for Creativity and Cognition

Submitted By




Semi Kağan Şahin - 2100005539

Timur Demir – 2200005944

Emre Koca - 2200003637










Department of Computer Engineering
İstanbul Kültür University
2024 
Analysis of EEG Data for Creativity and Cognition
 
 

ABSTRACT
This project aims to analyzing EEG signals recorded during developing of machine learning models to classify and predict cognitive states related to creativity and design tasks using EEG data from participants completing the Torrance Test of Creative Thinking (TTCT-F) creativity tasks to generate, evolve, and evaluate ideas. The primary objective is to identify patterns, trends, and differences in brain activity across these tasks. Using Python, various data analysis techniques were employed, including descriptive statistics, hypothesis testing (ANOVA), correlation analysis, and visualization methods like histograms, boxplots, and topographic maps. The results aim to contribute to understanding the neural correlates of creativity and resting states.

o	INTRODUCTION
A.	Problem Statement
The project investigates the differences in EEG signals between cognitive states (IDG, IDE, IDR) and resting (RST). It aims to explore whether the brain exhibits distinct patterns of activity during different cognitive tasks and whether resting state serves as a baseline for such comparisons.
B.	Project Purpose
The aim of this project is to analyze EEG data recorded during creativity tasks to reveal statistically significant patterns and correlations that can explain changes in brain activity, and to use this significant difference to train a model that can understand which task a person is doing from the EEG signal.
C.	Project Scope
•	The project includes 
•	 Pre-processing data set for each file (28 .mat files of different people)
•	 Extracting statistics based on the data on a person-by-person basis and on the basis of the whole experimental group and visualizing using this data. 
•	 Uncover differences between tasks, test a hypothesis about a model that can predict using these differences, and investigate the effect of rest states on tasks.

D.	Objectives and Success Criteria of the Project
•  We will calculate descriptive statistics and perform hypothesis tests (such as ANOVA) to assess task differences.
•  Create person-level and ensemble-level visualizations (histograms, boxplots, topographic maps) to interpret EEG signals.
• Provide insights into neural correlates of creativity.
E.	Report Outline
The report includes related work, the methodology used for analysis, experimental results, discussion, and conclusions.

o	RELATED WORK

A.	Existing Systems
Existing research often focuses on specific EEG frequency bands (e.g., alpha, beta) to study cognitive tasks. These studies use tools like MATLAB or specialized software for EEG analysis, but few provide a Python-based open-source approach.
B.	Overall Problems of Existing Systems
•  Limited focus on Python-based reproducible pipelines.
•  Insufficient visualization techniques to highlight task differences.
•  Complexity of setting up a structure that predicts differences between tasks.
C.	Comparison Between Existing and Proposed Method 
Table 2.1: Comparison of methods
Method	Data Source		Technique	Limitations
Band Analysis		EEG Data		Frequency Domain Analysis		Limited task comparison
Correlation Studies		EEG Data		Correlation Analysis		Poor task visualization
Proposed Method		Creativity Data		Python-based Statistical Tests and Visualization		Comprehensive analysis

o	METHODOLOGY
A.	OVERVİEW OF THE DATASET/MODEL
The dataset includes EEG recordings sampled at 500 Hz from 28 participants performing three creativity tasks (IDG, IDE, IDR) and two resting states (RST1, RST2). Each task and rest period has separate .mat files for each participant. Train a machine learning model to predict which class a signal belongs to.

B.	TOOLS AND TECHNOLOGY
•  Programming Language: Python
•  Libraries: Pandas, Matplotlib, Seaborn,Sklearn, SciPy, os, loadmat, welch
•  Models: Random Forest Classifier, Logistic Regression, XGBoost, Tensorflow
•  Environment: Jupyter Notebook, Matlab

PROPOSED APPROACH
•  Data Preprocessing:
Load .mat files to understand and read them.
Find and parse channels in the data
Use Welch library to reveal the 4 different bands within the 63 channels based on their frequency intensity.
•  Statistical Analysis:
Compute descriptive statistics (mean, standard deviation, skewness).
Perform ANOVA to test task differences.
Apply value counts to class labels inside the data.
•  Visualization:
Generate histograms, boxplots, and topographicmaps.
Compare RST with tasks visually and statistically.
Show the class distributions inside data.
Correlation matrix between 63 channels.
Show band density distributions inside data.

IV.	EXPERIMENTAL RESULTS
Descriptive Statistics: Number of data in each class; number, mean, standard deviation, min, max values were calculated for IDG, IDE, IDR groups for each project topic.

ANOVA Results: Statistically significant differences were observed between the IDE, IDG and IDR tasks themselves in different channels.

Visual Analysis:
Histograms showed the different data distributions for IDG, IDE and IDR, the differences in power between bands, and the differences between F1-precision-recall scores depends on classes.

Boxplots highlighted differences in signal amplitudes, frequency differences of bands within the signals, and frequency densities in classes within the data.
Correlation and Complexity matrices revealed differences and relationships between classes, as well as relationships between different channels.

Figures:
•	Histogram comparing data distributions between classes.
 ![image](https://github.com/user-attachments/assets/aa59a4fd-1e7c-4b7a-b4c1-b1d160fa5622)

•	Histogram comparing band powers inside data.
![image](https://github.com/user-attachments/assets/efdbff79-f703-4f57-8cce-dcd6d4388c09)

•	Histogram shows class-wise Precision, Recall and F1-Score
 ![image](https://github.com/user-attachments/assets/1144cfc0-4cd3-48f3-b871-5ddb1afabf46)

•	Histogram of Mean Signal Distributions (IDG, IDE, IDR): This histogram compares the distribution of mean signals across the three creative tasks: IDG (blue), IDE (green), and IDR (red). The differences in frequency distributions indicate task-specific variations in EEG activity. (Datas got from person 22, 26 and 20)
![image](https://github.com/user-attachments/assets/a80aea81-9735-4d9f-b265-1de01962955f)

    
Time series for each project subject emphasizes the fluctuations in EEG signal amplitude during the three creative tasks. The transitions between tasks are evident through signal 
changes. As a result for each project subject IDE, IDG and IDR groups have differences.

•	RST Time Series: The RST time series plot shows the EEG signal during resting state 1 (RST1) and resting state 2 (RST2) for Participant 7. The comparison highlights signal stability during resting periods. As the other plot (IDE, IDR, IDG comparison plot) RST1 ansd RST2 time series has differences for each participant.

![image](https://github.com/user-attachments/assets/9e0c6295-356b-4a1e-b94b-3382545f3980)

•	Channel comparison example from Creativity_9_1 key.
![image](https://github.com/user-attachments/assets/4145e7af-5e40-43bf-975a-eac5f947e309)

•	Boxplot Comparison (ch1_alpha_power): This boxplot shows ch1_alpha_power values for three task labels (IDE, IDG, IDR). Most values are very small and close to zero, with similar distributions for all labels. The middle line in each box shows the median. Some values are much higher (outliers), shown as circles above the boxes. The data looks a bit different for all three tasks. More tests are needed to find real differences.
![image](https://github.com/user-attachments/assets/5db6fd52-e03f-464b-8be5-cbfba17da6f0)

•	This is a pair plot showing relationships between three power features (ch1_delta_power, ch1_alpha_power, ch63_beta_power) for different task labels (IDE, IDG, IDR). The diagonal shows feature distributions, where most values are very small. The scatter plots compare features, showing some positive or nonlinear trends. Task labels are colored, but their distributions overlap a lot, meaning no clear separation. More analysis is needed to understand these patterns.
![image](https://github.com/user-attachments/assets/ad0c851e-c55a-4d1b-8728-96dba00c1447)

•	Correlation Matrix (IDG, IDE, IDR): This correlation matrix shows how features (such as delta_power, theta_power, alpha_power, and beta_power from different channels) are related. The diagonal values are 1 because each feature is perfectly correlated with itself. Red areas show strong positive correlations, and blue areas show strong negative correlations. Many features seem highly correlated, especially within the same type (e.g., alpha_power across channels). This suggests redundancy, and dimensionality reduction techniques like PCA might help simplify the data.
![image](https://github.com/user-attachments/assets/fe6bdf66-d12c-4fab-8f47-c8284ab96958)

•	PCA plot shows the distribution of EEG features for IDE, IDG, and IDR tasks. The points overlap a lot, meaning it is hard to separate the tasks clearly using the first two principal components (PC1 and PC2).
![image](https://github.com/user-attachments/assets/c6c2c297-863f-4fba-9e96-68c8236cf279)

 
•	We trained several different machine learning models, the most successful of which was the random forest classifier. We applied different methods to improve the performance of this model and you can see the effect of these methods on the success of this model.
![image](https://github.com/user-attachments/assets/e146fca2-d284-49ec-ac04-4ab73fac7111)

•	Confusion matrix shows the model's performance for predicting IDE, IDG, and IDR tasks. The model is good at predicting IDE (703 correct), but it struggles with IDG (329 incorrect) and IDR (115 incorrect). The results suggest the model needs improvement, especially for IDG and IDR predictions.
 ![image](https://github.com/user-attachments/assets/71bd14c6-be85-442b-b2e6-6398b694d47e)


V. DISCUSSION

•	Findings: Each participant has different signals when we look at tasks and resting states. The tasks (IDE, IDG, IDR) are quite different between each other in different time series and groups. EEG signal data gave different results in different ML models. We also found that there is an unbalanced distribution of data between 
classes in the dataset. To correct this, we used different data balancing methods and performed model training with these methods.
 
•	Interpretation: The most successful ML model applied to this experiment was the Random Forest Classifier. This model was the best predictor for our hypothesis with a success rate of 86%-90%. The Logistic Regression model and the XGboost model did not fit well with this model. We used RandomSearchCV and GridSearchCV methods to improve the XGBoost model, but the model accuracy still did not exceed 65-70%. We also tried to train the deep learning model Tensorflowa by applying CNN and LSTM techniques, but these rates were in the 50%-55% bands. This shows that the data we have is not complex enough to require the use of deep learning models. On the other hand, the class imbalance in our data also affected the performance of the models we trained. Data balancing methods: We used SMOTE, SMOTEEN, SMOTETomek, RandomUnderSampler, RandomOverSampler, RandomOverSampler methods. The most successful methods were RandomOverSampler and SMOTEEN, which we applied to our model, the Random Forest Classifier. These two methods improved the model accuracy by 86% and 90% while the others were between 79% and 86%.   
 
•	Potential Errors: We received EEG signal data in 28 different .mat file formats. If we made a mistake in this step, it may have affected the results of our experiment.
 
•	Applications: Insights from this study can help cognitive neuroscience research and the design of BCI systems. Also, whether increasing or decreasing differences between IDE, IDG and IDR classes cause a disease could be a topic for further research.



VI.		CONCLUSIONS
In this project, an experiment was conducted to interpret the EEG signals of 28 different people while performing 3 different tasks (IDE, IDG and IDR) and to look for significant differences between these signals. To fulfill this hypothesis, the data was processed, organized and visualized. Then, using certain methods, significant differences between them were revealed. Then, this significant difference was taught to a model and a model was trained to predict which task an EEG signal belongs to with a success rate of 86%-90%.

VII.		REFERENCES

• GitHub. (n.d.). GitHub Repository. Retrieved from https://github.com
• Kaggle. (n.d.).  https://www.kaggle.com/code/ruslankl/eeg-data-analysis
•  Stack Overflow. (n.d.). Stack Overflow - Where Developers Learn, Share, & Build Careers. Retrieved from https://stackoverflow.com
•  MNE Python. (n.d.). MNE: Open-source Python software for exploring, visualizing, and analyzing human neurophysiological data. Retrieved from https://mne.tools
•Mendeley Data. (n.d.). EEG Creativity Dataset. Retrieved from https://data.mendeley.com














