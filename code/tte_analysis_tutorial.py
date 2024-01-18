import os
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from lifelines.utils import median_survival_times
from lifelines.statistics import proportional_hazard_test
from lifelines import WeibullFitter, ExponentialFitter, LogNormalFitter, LogLogisticFitter


DATA_PATH = "../data/"

# Read data from NCCTG Lung Cancer Data. Survival in patients with advanced lung cancer from the North Central Cancer Treatment Group. Performance scores rate how well the patient can perform usual daily activities.

# Variables:
# inst:	Institution code
# time:	Survival time in days
# status:	censoring status 1=censored, 2=dead CHANGE TO  0=censored, 1=dead
# age:	Age in years
# sex:	Male=1 Female=2 CHANGE TO Male=0 Female=1
# ph.ecog:	ECOG performance score as rated by the physician. 0=asymptomatic, 1= symptomatic but completely ambulatory, 2= in bed <50% of the day, 3= in bed > 50% of the day but not bedbound, 4 = bedbound
# ph.karno:	Karnofsky performance score (bad=0-good=100) rated by physician
# pat.karno:	Karnofsky performance score as rated by patient
# meal.cal:	Calories consumed at meals
# wt.loss:	Weight loss in last six months
# Rows:
# 167
# Columns:
# 11

# Load data
data = pd. read_csv(os.path.join(DATA_PATH, "NCCTG_Lung_Cancer_Data_535_29.csv"))

# Filter and relabel data for improving comprehensability
data = data[["time", "status", "age", "sex", "ph.ecog", "ph.karno", "meal.cal", "wt.loss"]]
data["status"] = data["status"] - 1
data["sex"] = data["sex"] - 1

print(data.head())
print(data.describe())
print(data.dtypes)
print(data.isnull().sum()) # Check for missing values

# Plot Time data into histogram
T = data["time"] # days
E = data["status"] # event/death

plt.hist(T, bins=50)
plt.show()

#######################################################
# Fitting a non-parametric model [Kaplan Meier Curve] #
#######################################################
kmf = KaplanMeierFitter()
kmf.fit(durations=T, event_observed=E)
kmf.plot_survival_function() # Plot KM Estimate Survival function with confidence interval
plt.show()
kmf.survival_function_.plot() # Plot KM Estimate Survival function without confidence interval
plt.show()
kmf.plot_cumulative_density()
plt.show()
median_ = kmf.median_survival_time_
median_condifence_interval = median_survival_times(kmf.confidence_interval_)
print("Median survival time (50% probability of survival): " + str(kmf.median_survival_time_) + " days")
print("With 95% confidence intervals of : " + str(median_condifence_interval))

# Compare groups / segregate (i.e. gender)
ax = plt.subplot(111)
m = (data["sex"] == 0) # create mask where data (sex) = 0 (male)
kmf.fit(durations=T[m], event_observed=E[m], label="Male")
kmf.plot_survival_function(ax=ax)
kmf.fit(durations=T[~m], event_observed=E[~m], label="Female") # "~" will exclude the ones thta are true (i.e. select only the female)
kmf.plot_survival_function(ax=ax, at_risk_counts=True)
plt.title("Survival of different gender group")
plt.show()

# Automate segregation if more than 2 classes with a for loop
ecog_types = data.sort_values(by=["ph.ecog"])["ph.ecog"].unique() # We just want to find the unique values of the column, but we sort it first to have them in order

for i, ecog_type in enumerate(ecog_types):
    # ax = plt.subplot(int(len(ecog_types)/2), int(len(ecog_types)/2), i + 1) # each curve in different plots
    ax = plt.subplot(111) # all curves in one plot
    ix = data["ph.ecog"] == ecog_type # indices/mask of the rows with the chosen ecog_types
    kmf.fit(durations=T[ix], event_observed=E[ix], label=str(ecog_type))
    kmf.plot_survival_function(ax=ax)
    plt.title(ecog_type)
    plt.xlim(0, data["time"].max())

plt.title("Survival of different ECOG performance score group")
plt.tight_layout()
plt.show()


###################################################################
# Fitting a semi-parametric model [Cox Proportional Hazard Model] #
###################################################################

print(data.head())
# if we have more than 2 categories, we usually dummy encode those categories for analysis purposes
dummies_ecog = pd.get_dummies(data["ph.ecog"], prefix="ecog")

# compare ecog 1 and ecog 2
dummies_ecog = dummies_ecog[["ecog_1", "ecog_2"]] # extract 1 and 2
data = pd.concat([data, dummies_ecog], axis=1)    # concatenate with original data column-wise
data = data.drop("ph.ecog", axis=1)    # drop original ph.ecog column (redundant)

# Fit Cox model
cph = CoxPHFitter()
cph.fit(data, duration_col="time", event_col="status")
cph.print_summary()

# Interpretation

# In survival analysis, the p-value and -log2(p) are both measures of statistical significance for individual covariates in a Cox proportional hazards model. The Cox model is a popular statistical method for analyzing time-to-event data, such as survival data. It allows researchers to study the relationship between a set of predictor variables, called covariates, and the risk of an event, such as death or disease progression.

#The p-value is the probability of observing a test statistic as extreme or more extreme than the observed value, assuming the null hypothesis is true. In the context of Cox regression, the null hypothesis is that the covariate does not have a significant effect on the hazard ratio (the relative risk of the event occurring). A lower p-value indicates that the observed association between the covariate and the event is more li# kely to be due to a real effect rather than chance.

# The -log2(p) is a transformation of the p-value that is often used in survival analysis because it has a more intuitive interpretation. The -log2(p) value is a measure of the strength of the association between the covariate and the event. A higher -log2(p) value indicates a stronger association, and a negative -log2(p) value indicates that the covariate is associated with a decreased risk of the event.

# In general, a p-value of less than 0.05 is considered statistically significant, indicating that there is a strong evidence that the covariate is associated with the event. However, the interpretation of the -log2(p) value is more complex and can depend on the specific context of the study.

# plot dependence coefficeint
plt.subplots(figsize=(10,6))
cph.plot()
plt.show()

cph.plot_partial_effects_on_outcome(covariates="age", values=[50,60,70,80], cmap="coolwarm")
plt.show()

# check assumptions
cph.check_assumptions(data, p_value_threshold=0.05)

# proportional hazard test
results = proportional_hazard_test(cph, data, time_transform="rank")
results.print_summary(decimals=3, model="untransformed variables")


###################################################################
# Fitting parametric model [Accelerated Failure Time Model (AFT)] #
###################################################################

print("Fitting parametric model [Accelerated Failure Time Model (AFT)")
# instantiate each fitter
wb = WeibullFitter() # Weibull distribution
ex = ExponentialFitter()
log = LogNormalFitter()
loglogis = LogLogisticFitter()

# Fit to data
for model in [wb, ex, log, loglogis]:
    model.fit(durations=data["time"], event_observed=data["status"])
    # print AIC
    print("The AIC value for ", model.__class__.__name__, " is ", model.AIC_)

#
# The Akaike information criterion (AIC) is a statistical measure used to compare different parametric models for a given set of data. It evaluates the trade-off between a model's goodness-of-fit and its complexity. A lower AIC value indicates a better model.

# The Weibull fitter is best
from lifelines import WeibullAFTFitter
weibull_aft = WeibullAFTFitter()
weibull_aft.fit(data, duration_col="time", event_col="status")
weibull_aft.print_summary()

# intrepretations of the coeficients
# sex has a positive coefficient
# this means being a female subject compared to male cjanges the mean/median survival time by exp(0.41) = 1.50, approximately a 52% increase in mean/median survival time.
print("Weibull Mean survival time (50% probability of survival): " + str(weibull_aft.mean_survival_time_) + " days")
print("Weibull Median survival time (50% probability of survival): " + str(weibull_aft.median_survival_time_) + " days")

plt.subplots(figsize=(10, 6))
weibull_aft.plot()
plt.show()

plt.subplots(figsize=(10,6))
weibull_aft.plot_partial_effects_on_outcome("age", range(50, 80, 10), cmap="coolwarm")
plt.show()

plt.subplots(figsize=(10,6))
weibull_aft.plot_partial_effects_on_outcome("ph.ecog", range(1000, 2500, 500), cmap="coolwarm")
plt.show()

print("hola")



