#!/usr/bin/env python
# coding: utf-8

# In[1]:


# numpy is used for numerical computations and array manipulation
#  pandas is used for data manipulation and analysis
import pandas as pd
import numpy as np

df = pd.read_csv('NY-House-Dataset.csv')
df.head()


# 
# # Data Exploration

# In[2]:


df.shape


# In[3]:


df.info()


# In[4]:


# This code iterates through each column of a DataFrame (df) and prints the column
# name followed by the count of unique values in that column, separated by dashes.

for i in df:
  print(i)
  print(df[i].value_counts())
  print("----------")


# ##As we see a lot of objects column and that might be useful for our model, we are gonna possible apply all the approach to make them useful

# # Data Visualization

# In[5]:


import plotly.express as px
scatter_plot_multicolor_log = px.scatter(df, x='PROPERTYSQFT', y='PRICE', color='TYPE',
                                          title='Price vs. Property Square Footage (Logarithmic Scale)',
                                          labels={'PROPERTYSQFT': 'Property Square Footage', 'PRICE': 'Price', 'TYPE': 'House Type'},
                                          log_x=True, log_y=True)
scatter_plot_multicolor_log.show()


# In[6]:


box_plot_log = px.box(df, x='TYPE', y='PRICE', color='TYPE' ,title='Price Distribution by House Type (Logarithmic Scale)',
                      labels={'TYPE': 'House Type', 'PRICE': 'Price'},
                      log_y=True)
box_plot_log.show()


# In[7]:


scatter_3d_multicolor = px.scatter_3d(df, x='BEDS', y='BATH', z='PRICE', title='Price vs. Bedrooms vs. Bathrooms',
                                       labels={'BEDS': 'Bedrooms', 'BATH': 'Bathrooms', 'PRICE': 'Price'},
                                       color='PRICE')
scatter_3d_multicolor.show()


# In[8]:


sunburst_chart = px.sunburst(df, path=['STATE', 'LOCALITY', 'SUBLOCALITY'], title='Categorization of Houses')
sunburst_chart.show()


# # Data Preprocessing

# In[9]:


df.head()


# In[10]:


df.columns


# In[11]:


# Counting the occurrences of each broker title
broker_counts = df['BROKERTITLE'].value_counts()

# Identifying the top 10 brokers
top_brokers = broker_counts.nlargest(10).index

# Replacing all other broker titles with 'Other'
df['BROKERTITLE_REDUCED'] = df['BROKERTITLE'].apply(lambda x: x if x in top_brokers else 'Other')

df.drop(['BROKERTITLE'],axis=1,inplace=True)

# Checking the modification
print(df['BROKERTITLE_REDUCED'].value_counts())


# ## Now for TYPE Column

# In[12]:


# Checking unique values in the "TYPE" column
type_unique_values = df['TYPE'].unique()
type_unique_values


# ### Now for address column

# ## There is a very high cardinality in this column so we are going to drop it

# In[13]:


df.drop(['ADDRESS'],axis=1,inplace=True)


# ## Now for STATE column

# In[14]:


print(df['STATE'].value_counts().to_string())


# In[15]:


# List of New York City boroughs
nyc_boroughs = ["FLoral Park","East Elmhurst","College Point","Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island",'Fresh Meadows',"Bellerose","Glen Oaks","Hollis","Jamaica","Rosedale","Douglaston","Astoria","Averne","Queens","Forest Hills"]

# Function to find the borough name in the given text
def extract_borough(text):
    # Normalize text to handle different case entries
    text = text.lower()
    # Search for any borough name present in the text
    for borough in nyc_boroughs:
        if borough.lower() in text:
            return borough
    return "Other"  # Return 'Unknown' if no borough is found

# Apply the function to the 'STATE' column to create a new column 'Borough'
df['Borough'] = df['STATE'].apply(extract_borough)

# Verify the results by showing the unique values in the new column
unique_boroughs = df['Borough'].unique()
print(unique_boroughs)

df['Borough'].value_counts()


# ### Have listed all the boroughs that are quite common in the state column and made a new column with just the boroughs and rest keeping as unkown

# ### Now for extracting postal codes from State column

# In[16]:


# Function to find the postal code in the given text
import re
def extract_postal_code(text):
    match = re.search(r'\b\d{5}\b', text)
    return match.group(0) if match else "Unknown"

# Apply the function to the 'STATE' column to create a new column 'Postal_Code'
df['Postal_Code'] = df['STATE'].apply(extract_postal_code)

# Verify the results by showing the unique values in the new column
unique_postal_codes = df['Postal_Code'].unique()
unique_postal_codes


# now droppig the column MAIN_ADDRESS and going ahead with the extraction of ADMINISTRATIVE_AREA_LEVEL_2 column

# The "ADMINISTRATIVE_AREA_LEVEL_2" column in the dataset contains 29 unique values. These values appear to be a mix of county names, postal codes, parts of New York City like "Brooklyn," and some general entries like "United States. so lets drop that as well

# In[17]:


df.drop(['STATE','MAIN_ADDRESS','ADMINISTRATIVE_AREA_LEVEL_2'],axis=1,inplace=True)


# In[18]:


df.info()


# 
# Now for localities

# In[19]:


# Checking unique values and their frequency in the 'LOCALITY' column
locality_counts = df['LOCALITY'].value_counts()
locality_counts


# already categorical not much needed

# the next column SUBLOCALITY has already captured with some prior info so lets drop it

# In[20]:


df.drop(['SUBLOCALITY'],axis=1,inplace=True)


# In[21]:


df.info()


# In[22]:


print(df['STREET_NAME'].value_counts().to_string())


# In[23]:


# keep the top 6 street names with highest value counts and rest as other

top_street_names = df['STREET_NAME'].value_counts().nlargest(6).index

df['STREET_NAME_REDUCED'] = df['STREET_NAME'].apply(lambda x: x if x in top_street_names else 'Other')

df.drop(['STREET_NAME'],axis=1,inplace=True)


# In[24]:


df['STREET_NAME_REDUCED'].value_counts()


# Done with Street names

# In[25]:


df.info()


# In[26]:


df['LONG_NAME'].value_counts()


# dropping it due to high cardinality and very low relevance

# In[27]:


df.drop(['LONG_NAME'],axis=1,inplace=True)


# In[28]:


df.info()


# In[29]:


df['FORMATTED_ADDRESS'].value_counts()


# dropping it out as well

# In[30]:


df.drop(['FORMATTED_ADDRESS'],axis=1,inplace=True)


# In[31]:


df.info()


# keep the Latitude and longitude will provide specific geographic locations that allow the model to capture spatial relationships. Properties that are geographically close to each other tend to have similar prices due to shared environmental and economic factors.

# In[32]:


import seaborn as sns
sns.boxplot(df['PRICE'])


# In[33]:


df[df['PRICE']>10000000]


# In[34]:


df.drop(df[df['PRICE']>10000000].index,inplace=True)


# In[35]:


import seaborn as sns
sns.boxplot(df['PRICE'])


# In[36]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['PRICE_standardized'] = scaler.fit_transform(df[['PRICE']])
sorted_df = df.sort_values(by='PRICE_standardized', ascending=False)
sorted_df = sorted_df[2:]


# In[37]:


df.isnull().sum()


# In[38]:


df.duplicated().sum()


# In[39]:


df.drop_duplicates(inplace=True)


# In[40]:


df.info()


# In[41]:


df['BATH'] = df['BATH'].astype(int)
df['PROPERTYSQFT'] = df['PROPERTYSQFT'].astype(int)
df['LATITUDE'] = df['LATITUDE'].astype(int)
df['LONGITUDE'] = df['LONGITUDE'].astype(int)
df['PRICE_standardized'] = df['PRICE_standardized'].astype(int)


# In[42]:


from sklearn.preprocessing import LabelEncoder
# Identify object columns
object_cols = df.select_dtypes(include=['object']).columns
print("Object columns:", object_cols)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each object column
for col in object_cols:
    df[col] = label_encoder.fit_transform(df[col])


# In[43]:


# Verify the changes
print(df.head())
print(df.dtypes)


# In[44]:


df.info()


# ## Save cleaned csv file 

# In[45]:


# # Save the cleaned DataFrame to a CSV file
# cleaned_file_path = 'cleaned_housedata.csv'
# df.to_csv(cleaned_file_path, index=False)


# ## Training and Testing

# In[46]:


X = df[['BEDS', 'BATH', 'PROPERTYSQFT', 'LOCALITY', 'Borough', 'Postal_Code', 'STREET_NAME_REDUCED']]
y = df['PRICE']


# In[47]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[48]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)


# In[49]:


from sklearn.preprocessing import PolynomialFeatures

# Initialize polynomial features
poly = PolynomialFeatures(degree=4, include_bias=False)

# Fit and transform the data
X_poly = poly.fit_transform(X)


# ## Modelling
# ### 1. LinearRegression

# In[50]:


from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)


# In[51]:


# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred


# In[52]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("r2_score is: ", r2)
print("mse is: ", mse)
print("mae is: ", mae)


# ### 2.  RandomForestRegressor

# In[53]:


from sklearn.ensemble import RandomForestRegressor

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)


# In[54]:


# Make predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_rf


# In[55]:


# Evaluate the model
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("Random Forest r2_score is: ", r2_rf)
print("Random Forest mse is: ", mse_rf)
print("Random Forest mae is: ", mae_rf)


# ## Predict the New Data 

# In[56]:


# Predict house price for a new data point
new_data_point = pd.DataFrame([[10,3, 5000, 0,0,0,0]], 
                        
                  columns=['BEDS', 'BATH', 'PROPERTYSQFT', 'LOCALITY', 'Borough', 'Postal_Code', 'STREET_NAME_REDUCED'])

predicted_price = model.predict(new_data_point)
print(f"Predicted Price for the new data point: ${predicted_price[0]}")

