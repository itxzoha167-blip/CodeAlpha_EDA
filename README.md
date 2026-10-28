# CodeAlpha Task 2: Exploratory Data Analysis (EDA)

## Project Overview
This project performs Exploratory Data Analysis (EDA) on the Students Performance in Exams dataset. The goal is to understand student score distributions, compare academic performance across groups, identify potential outliers, and examine the relationship between test preparation and mathematics scores.

## Dataset
The dataset contains 1,000 student records and 8 columns, including:
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Math Score
- Reading Score
- Writing Score

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy

## Analysis Performed
- Dataset structure and data types
- Missing-value and duplicate checks
- Score range validation
- Descriptive statistics
- Average scores by gender
- Score comparison by test preparation status
- Potential outlier detection using the IQR method
- Welch's independent t-test for mathematics scores
- Data visualization

## Key Findings
- The dataset contains 1,000 records and 8 columns.
- No missing values or duplicate rows were detected.
- Average mathematics score was approximately 66.09.
- Students who completed the test preparation course had a higher average mathematics score (69.70) than students who did not (64.08).
- The statistical test found evidence of a difference in average mathematics scores between the two groups at the 5% significance level.

These findings show an association in this dataset; they do not prove that test preparation alone caused the score difference.

## Output Files
The `outputs` folder contains summary statistics, outlier analysis, hypothesis-test results, visualizations, and a written analysis summary.

## How to Run
1. Install Python.
2. Install the required libraries:
   `python -m pip install pandas numpy matplotlib seaborn scipy`
3. Place `StudentsPerformance.csv` in the project folder.
4. Run:
   `python eda_analysis.py`

## Project
Completed as part of the CodeAlpha internship task.
