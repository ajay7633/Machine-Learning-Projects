![](/Blight%20Ticket%20Compliance/images/detroit.png)


# PROJECT - BLIGHT TICKET COMPLIANCE

## Overview
This project is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 

The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city who allow their properties to remain in a deteriorated condition. The city of Detroit gives millions of dollars in fines to residents, and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know how to increase blight ticket compliance?
The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. 

In this project we predict whether a given blight ticket will be paid on time or not.

We provide you with two data files for training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing date, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance and a handful of other variables that will not be available at test-time are only included in train.csv.

## Data

train.csv & test.csv

    ticket_id - unique identifier for tickets
    agency_name - Agency that issued the ticket
    inspector_name - Name of inspector that issued the ticket
    violator_name - Name of the person/organization that the ticket was issued to
    violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
    mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
    ticket_issued_date - Date and time the ticket was issued
    hearing_date - Date and time the violator's hearing was scheduled
    violation_code, violation_description - Type of violation
    disposition - Judgment and judgement type
    fine_amount - Violation fine amount, excluding fees
    admin_fee - $20 fee assigned to responsible judgments
state_fee - $10 fee assigned to responsible judgments
    late_fee - 10% fee assigned to responsible judgments
    discount_amount - discount applied, if any
    clean_up_cost - DPW clean-up or graffiti removal cost
    judgment_amount - Sum of all fines and fees
    grafitti_status - Flag for graffiti violations
    
train.csv only

    payment_amount - Amount paid, if any
    payment_date - Date payment was made, if it was received
    payment_status - Current payment status as of Feb 1 2017
    balance_due - Fines and fees still owed
    collection_status - Flag for payments in collections
    compliance [target variable for prediction] 
     Null = Not responsible
     0 = Responsible, non-compliant
     1 = Responsible, compliant
    compliance_detail - More information on why each ticket was marked compliant or non-compliant


___

## Evaluation

The predictions will be given as the probability that the corresponding blight ticket will be paid on time.
The evaluation metric for this project is the Area Under the ROC Curve (AUC). 

*  Did NOT use Neural Network related classifiers (e.g., MLPClassifier)

## Setup

We only take consider features in the test data columns so all common features are cosidered and rest are dropped & not considered

From the Test Data : Some of features from initial consideration are as follows:

Numerical Data:

 1. Judgement amount ( Total Net Amount owed by person)
 3. Late fee         (If any late fee incurred by the poi - person of interest)
 4. Fine amount      (Original Amount)
 P.S: All the fees like(state fees,admin fee etc are dropped cause its standard fee and does not help with prediction)
Categorical Data:

 This can have an impact on the prediction, as the person from a 'certain place' might not consider paying
 and can have better understanding on prediction , atleast provide understanding if people are paying in certain   locations than others

 1. City
 2. State
 3. Pincode (Yet to decide)
 4. Disposition
3.We try to choose features that dont blow up in to 100s and 1000s of features after creating dummy variables.

#### Part 1: Loading the data
#### Part 2: Cleaning up the data
  1.Removing unwanted rows (NA values) - Training Data ,  Test Data should not have any rows dropped.
  2.Dropping unnecessary columns that doesnt help with predictions - For Both Test and Training Data.
#### Part 3: Creating Dummy variables (if required) and hot encoding categorical data
    1. We need to hot encode the disposition (it becomes numbered )
    2. Dont forget to do the same transformation in Test data
        - We are replacing disposition in to reflect both the Training and Test data
    3. Ziping up can make it easier to lookup that information (if you want)
We are going to consider the following columns for our classifier

 FOR X:

  1.(Disposition columns) - after hot encoding -4 columns
  2.Judgement Amount
  3.Late fee

 FOR Y:
  1.Compliance
#### Part 4 : Choosing a Classifier and testing the performace of data (HyperParameter Tuning)
 
 1. Logistic Regression
  2. LinearSVM
  3. Decision Tree
  4. Random Forest
  5. Naive Bayes

Finding out the best model with the best parameter to give best performace

I chose Logistic Regression and calculated auc score, best parameter and score

