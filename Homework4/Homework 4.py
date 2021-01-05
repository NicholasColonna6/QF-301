
# coding: utf-8

# # Homework 4
# ## Nicholas Colonna
# ### "I pledge my honor that I have abided by the Stevens Honor System"

# In[29]:


import pandas as pd
import numpy as np
import pylab as pl
import statsmodels as sm
import matplotlib.pyplot as plt
import math
import statistics as stat

df = pd.read_csv("./Boston.csv")
print(df.head())


# (a) Based on this data set, provide an estimate for the population mean of medv. Call this estimate ˆμ.
# 
# 
# The estimate for the population mean is calculated by simply taking the sum of all medv data points and dividing it by the number of observations. This gives us an estimate of 22.5328

# In[30]:


observations = df['medv'].count()

e_mean = sum(df.medv) / observations
print(e_mean)


# (b) Provide an estimate of the standard error (SE) of ˆμ. Interpret this result.
# Hint: We can compute the standard error of the sample mean by dividing the sample standard deviation by the square root of the number of observations.
# 
# I used the formula for standard deviation to arrive at an estimate for SE of 0.4084

# In[31]:


sum_square = 0
for val in df.medv:
    sum_square = sum_square + (val - e_mean)**2

sample_stdev = math.sqrt(sum_square/(observations))

e_stdev = sample_stdev / math.sqrt(observations)
print(e_stdev)


# (c) Now estimate the standard error of ˆμ using the bootstrap. How does this compare to your answer from (b)?
# Hint: You can use the following function to generate a bootstrap sample. Your solution should generate many bootstrap samples to calculate the standard error.
# 
# The calculaed standard error using 500 bootstrap samples gives a value around 0.41, which is fairly close to our calculated SE from our data in part b.

# In[38]:


def bootstrap_sample(df, column, statistic):
       """Compute bootstrap sample of column in dataframe df applying statistic function."""
       # get a random sampling of indices
       boot_indices = np.random.choice(df.index, size=len(df), replace=True)
       # compute a sample statistic
       sample_stat = statistic(df[column].loc[boot_indices])
       return sample_stat

bootstrap_stdev = []
for i in range(0,501):
   bootstrap_stdev.append((bootstrap_sample(df, 'medv', stat.stdev))/math.sqrt(observations))

bootstrap_stdev = np.mean(bootstrap_stdev)
print(bootstrap_stdev)


# (d) Based on your bootstrap estimate from (c), provide a 95% confidence interval for the mean of medv. Compare it to the results obtained using the t test of medv. Hint: You can approximate a 95% confidence interval using the formula: ˆμ − 2SE(ˆμ), ˆμ + 2SE(ˆμ).
# 
# First, I calculated the upper and lower bound of the 95% confidence interval, then I displayed it as an inequality. I also ran a t-test on the data, and generated a confidence interval. As you can see from the results, the two methods yield very similar results.

# In[39]:


upper = e_mean + 2*bootstrap_stdev
lower = e_mean - 2*bootstrap_stdev
print("Bootstrap 95% Confidence Interval: ", lower, " < μ < ", upper, sep='')

t_test = sm.stats.weightstats.DescrStatsW(df.medv)
t_test_ci = t_test.tconfint_mean()
print("   t-test 95% Confidence Interval: ", t_test_ci[0], " < μ < ", t_test_ci[1], sep='')


# (e) Based on this data set, provide an estimate, ˆμmed, for the median value of medv in the population.
# 
# Since we have an even number of observations (506), the median will be the average between the two middle-most data points (253 and 254). Using the statistics median function, we obtain a median of 21.2.

# In[40]:


e_median = stat.median(df.medv)
print(e_median)


# (f) We now would like to estimate the standard error of ˆμmed. Unfortunately, there is no simple formula for computing the standard error of the median. Instead, estimate the standard error of the median using the bootstrap. Comment on your findings.
# 
# Using the bootstrap_sample function above, I called the function many times to generate many bootstrap samples.

# In[41]:


bootstrap_median = []
for i in range(0,501):
    bootstrap_median.append(bootstrap_sample(df, 'medv', stat.median))

bootstrap__med_stdev = stat.stdev(bootstrap_median)
print(bootstrap__med_stdev)


# (g) Based on this data set, provide an estimate for the tenth percentile of medv in Boston suburbs. Call this quantity ˆμ_0.1.
# 
# Using the numpy percentile function, I was able to obtain the 10th percentile of the data, which is 12.75.

# In[42]:


tenth_percentile = np.percentile(df['medv'], 10)
print(tenth_percentile)


# (h) Use the bootstrap to estimate the standard error of ˆμ_0.1. Comment on your findings
# 
# I modified the bootstrap function slightly to take in one additional parameter, which indicates the percentile we want. In addition, I had to adjust the sample_stat line to include the new variable percentile. From there, it follows the same method as previous examples, creating multiple bootstrap samples and then calculating the standard deviation from the results.

# In[43]:


def bootstrap_sample_percentile(df, column, statistic, percentile):
        """Compute bootstrap sample of column in dataframe df applying statistic function."""
        # get a random sampling of indices
        boot_indices = np.random.choice(df.index, size=len(df), replace=True)
        # compute a sample statistic
        sample_stat = statistic(df[column].loc[boot_indices], percentile)
        return sample_stat
    
    
bootstrap_tenth = []
for i in range(0,501):
    bootstrap_tenth.append(bootstrap_sample_percentile(df, 'medv', np.percentile, 10))

bootstrap_tenth_stdev = stat.stdev(bootstrap_tenth)
print(bootstrap_tenth_stdev)

