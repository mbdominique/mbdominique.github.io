#!/usr/bin/env python
# coding: utf-8

# # Exploring Uber and Lyft Prices in Boston

# ## Dominique Bradshaw
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utility.util import configure_plots
configure_plots()
#all of the things I think we could need here, hmm


# The data comes from  Uber and Lyft in Boston, MA  from 11-26-2018 to 12-18-2018.How the information was obtained was not dsiclosed Our data has 693071 data points and 57 features. We have mostly non-catgeorical data types. The values of these features will depend on the data itself. For example, the "Trip date and time" feature may contain timestamps, while the "Pickup and drop-off locations" feature would have location data. Other features, like "Fare amount" and "Distance," would contain numerical values.The features consists of ID, time-related aspects(timestamp, hour, day, month, etc.) of the day, the settings of the day (temperatures, expected temperatures, tempMax, tempMin), locations, basically anything related to the settings of the day of the lyft/uber ride.
# 

# In[5]:


#Load raw data 
rideshare= pd.read_csv("./utility/data/rideshare_kaggle.csv")
# Exploring the data
features = rideshare.columns
print(features)
rideshare.head()


# In[6]:


rideshare.count()


# In[7]:


rideshare.describe()


#  
# 
# ### **Wait, is there something missing** That's right, let's check for missing values, in this case. Let's go ahead and drop them. If our dataset was signifcantly smaller, I would opt for imputing mean values.

# Steps: We need to decide and categorize the kinds of data types, like between categorical and non-categorical and possbily what features we should not use due to ethical reasons 

# In[8]:


# Check for missing values
missing_values = rideshare.isnull().sum()
print("Missing Values:\n", missing_values)


# In[9]:


# Handle missing values, in this case, we will drop them
#rideshare['price'].fillna(rideshare['price'].mean(), inplace=True)
rideshare.dropna(subset=['price'], inplace=True)


# In[10]:


# Remove duplicates
rideshare.drop_duplicates(inplace=True)


# In[11]:


# Price Distribution by Ride Type
plt.figure(figsize=(10, 6))
sns.violinplot(data=rideshare, x='cab_type', y='price')
plt.xlabel('Ride Type')
plt.ylabel('Price')
plt.title('Price Distribution vs Ride Type')
plt.xticks(rotation=45)
plt.show()


# In[12]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=rideshare, x='cab_type', y='price')
plt.xlabel('Ride Type')
plt.ylabel('Price')
plt.title('Price Distribution vs Ride Type')
plt.xticks(rotation=45)
plt.show()


# In[13]:


plt.figure(figsize=(10, 6))
sns.lineplot(data=rideshare, x='precipIntensity', y='surge_multiplier', alpha=0.5)
plt.xlabel('Precipitation Intensity')
plt.ylabel('Surge Multiplier')
plt.title('Precipitation Intensity vs. Surge Multiplier')
plt.ylim(0, 1.5)
plt.show()


# ## Hypothesis: Uber fares are expected to consistently exhibit a more cost-effective pricing structure in comparison to Lyft fares, irrespective of environmental variables such as weather conditions and time of day or month.
# 
# Our research focus is oriented towards the examination of pricing disparities inherent to each ride-sharing service, stemming from factors such as quality, operational policies, and associated fees.
# 
# Our empirical analysis will discern the overall pricing discrepancies across the spectrum of ride-sharing services, independent of the influences of factors such as distance, meteorological conditions, and temporal considerations. This, in turn, will enable us to provide discerning customers with insights into the underlying determinants of price differentiation, empowering them to make informed choices in accordance with their budgetary preferences.

# # Let's Work With This Hypothesis, HMM, was it on par?

# In[21]:


uber_data = rideshare[rideshare['cab_type'] == 'Uber']
lyft_data = rideshare[rideshare['cab_type'] == 'Lyft']

X_uber = uber_data[['distance', 'temperature', 'humidity']]
y_uber = uber_data['price']
X_uber_train, X_uber_test, y_uber_train, y_uber_test = train_test_split(X_uber, y_uber, test_size=0.3, random_state=11)

X_lyft = lyft_data[['distance', 'temperature', 'humidity']]
y_lyft = lyft_data['price']
X_lyft_train, X_lyft_test, y_lyft_train, y_lyft_test = train_test_split(X_lyft, y_lyft, test_size=0.3, random_state=11)

average_price_uber = np.mean(y_uber_train)
average_price_lyft = np.mean(y_lyft_train)
print(average_price_uber)
print(average_price_lyft)


# # Let's try a super basic Linear Regression

# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

uber_model = LinearRegression()
lyft_model = LinearRegression()

uber_model.fit(X_uber_train, y_uber_train)
lyft_model.fit(X_lyft_train, y_lyft_train)

uber_predictions = uber_model.predict(X_uber_test)
lyft_predictions = lyft_model.predict(X_lyft_test)

uber_mae = mean_absolute_error(y_uber_test, uber_predictions)
lyft_mae = mean_absolute_error(y_lyft_test, lyft_predictions)

uber_mse = mean_squared_error(y_uber_test, uber_predictions)
lyft_mse = mean_squared_error(y_lyft_test, lyft_predictions)

uber_r2 = r2_score(y_uber_test, uber_predictions)
lyft_r2 = r2_score(y_lyft_test, lyft_predictions)

print("Uber Model:")
print(f"Mean Absolute Error: {uber_mae:.2f}")
print(f"Mean Squared Error: {uber_mse:.2f}")
print(f"R-squared (R2): {uber_r2:.2f}")

print("\nLyft Model:")
print(f"Mean Absolute Error: {lyft_mae:.2f}")
print(f"Mean Squared Error: {lyft_mse:.2f}")
print(f"R-squared (R2): {lyft_r2:.2f}")


# # Now Let's Try a Decision Tree, linear might not be a good fit :(

# In[24]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train the Decision Tree model for Uber
uber_tree = DecisionTreeRegressor(random_state=11)
uber_tree.fit(X_uber_train, y_uber_train)

# Predict and evaluate Uber model
uber_tree_predictions = uber_tree.predict(X_uber_test)
print("Uber Decision Tree MAE:", mean_absolute_error(y_uber_test, uber_tree_predictions))
print("Uber Decision Tree MSE:", mean_squared_error(y_uber_test, uber_tree_predictions))
print("Uber Decision Tree R2:", r2_score(y_uber_test, uber_tree_predictions))

# Train the Decision Tree model for Lyft
lyft_tree = DecisionTreeRegressor(random_state=11)
lyft_tree.fit(X_lyft_train, y_lyft_train)

# Predict and evaluate Lyft model
lyft_tree_predictions = lyft_tree.predict(X_lyft_test)
print("Lyft Decision Tree MAE:", mean_absolute_error(y_lyft_test, lyft_tree_predictions))
print("Lyft Decision Tree MSE:", mean_squared_error(y_lyft_test, lyft_tree_predictions))
print("Lyft Decision Tree R2:", r2_score(y_lyft_test, lyft_tree_predictions))


# # This is still pretty bad, but with more practice (with your help). I could create wonderful models and strengthen my skills. Thank you for taking the time to look at my project.

# The average price of Uber in relation to distance is $15.77 where as the average price of Lyft in relation to distance is $17.36. There is a $1.59 price difference between the two cabs. While this may seem indifferent to some customers, for customers who use the transportation services regularly, this price adds up. In a day, that is an additional $3.18 for one ride back and forth, $6.36 for two rides back an forth. In a week, $22.26 for one ride back and forth each day, $44.52 for two rides back and forth each day. In a month, around $95 for one ride back and forth for 30 days, around $190 for two rides back and forth for 30 days. The price adds up and for areas where many people do not own cars such as Chicago, New York, and other big cities, in addition to the high cost of living, customers would be interested in spending less.
# 
# Also, from Uber's dataset, there is a lower Mean Absolute Error of 6.69 than Lyft's dataset in which the Mean Absolute Error is 7.39. Therefore, the machine learning model for Uber has less errors than the machine learning model for Lyft.
# 
# Uber has a smaller Mean Squared Error of 64.96 in comparison to Lyft's Mean Squared Error of 86.86. This means that Uber's estimates are closer to the actual values than Lyft's, supporting the idea that Uber's prices are better than Lyft's.
# 
# Surprisingly, the Lyft regression model fits the data better and has a better performance with 0.13 in comparison to Uber's regression model fo 0.11. Although this may indicate a better fit, both values are fairly low, leaving a conclusion that the model may not actually explain any of the variability in the dependent variables. Also, a low R-squared could indicate that the model may not actually be a good fit for the data.
# 

# ### Ethical Issues, Who Holds Stake?

# Stakeholders: The manufacturers of the models that calculate the prices for riders/customers are stakeholders that do the affecting. The drivers/employees are affected by the model/app as the prices may determine how many rides they give in a day, affecting their wages. The riders/customers are affected by the prices of the rides as well as the actual experience of the rides. If drivers aren't being paid a livable wage, there will be a decrease in supply and an increase in demand in which there will be less drivers and a disproportionate amount of riders/customers to each driver.
#  
# Impacted by ethical issues: One of the ethical issues is the surging prices of the services. With the reliance on transportation systems in big cities, especially New York, surging prices can seem exploitative as other transportation systems (trains, buses, etc.) usually have fixed prices. With the surging of prices and need for transportation in big cities where most citizens don't have cars and high traffic, high prices/price surges may seem like companies like Lyft with higher prices are taking advantage of its customers.
# 
# Resolution: A resolution to the exploitation of prices is to set a fixed price for distance as well as a fixed price for time of day. Also, if companies consult each other rather than focusing on beating competition. This would demostrate fairer practices, which would increase the demand of their transportation services.

# # Future Improvements

# The features that were helpful were source, destination, distance (needed source and destination to get the distance), cab type, and price were informative in our analysis of relationships as we rarely found meaningful relationships between weather (precipitation, UVIndex, visibility, wind, temperature, etc.). Features that were not informative were features related to weather, sunrise, sunset, moonphase, temperature, etc.) because of the lack of relationships that we found with prices, price surges, and more. it would be helpful to have the volume of rides for each day, week, and month so we can determine if there is a relationship between the non-informative features as well as the informative features to see if the volume of rides are affected in any way.
