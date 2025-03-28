#1. Basic Python Programming: 
# Write a "Hello, World!" program. 
# Practice basic data types, variables, and operators. 
print( "Hello, World!")
# Declare variables with different data types
name = "Aviral"         # String
age = 18               # Integer
height = 5.6           # Float
is_student = True      # Boolean

# Print the variables
print("Name:", name)
print("Age:", age)
print("Height:", height)
print("Is student?", is_student)

# Perform basic arithmetic operations
sum = 5+3
difference = 5-3
product = 5*3
division = 5/3

# Output the results of the operations
print("Sum:", sum)
print("Difference:", difference)
print("Product:", product)
print("Division:", division)

# 2. Control Structures: 
# Use loops (for, while) and conditionals (if, else) to solve simple problems.

#Sum of natural numbers from 1 to 100
sum=0
for i in range(1,101):
    sum=sum+i
print(sum)
# Using a while loop to print even numbers between 2 and 10
num = 2
while num <= 10:
    print(num)
    num += 2  # Increment by 2 to get the next even number

# Using if, elif, else to check if a number is positive, negative, or zero
number = int(input("Enter a number: "))

if number > 0:
    print("The number is positive.")
elif number < 0:
    print("The number is negative.")
else:
    print("The number is zero.")

# 3. NumPy Essentials: 
# Create and manipulate NumPy arrays. 
# Perform basic arithmetic operations and aggregations (mean, median, sum). 

import numpy as np

# Create a NumPy array from a list
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)

# Manipulation of array:

# Reshape an array (change dimensions)
reshaped = arr.reshape(5, 1)
print("Reshaped Array:\n", reshaped)

# Access specific elements
print("Third element is:", arr[2])

# Basic Arithmetic Operations:

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

add = arr1+arr2
print("Addition:",add)

sub = arr1-arr2
print("Subtraction:",sub)

mult = arr1*arr2
print("Multiplication:",mult)

div = arr1/arr2
print("Division:",div)

# Basic Aggregations:

# Mean:
mean = np.mean(arr1)
print("Mean:", mean)

# Median
median = np.median(arr1)
print("Median:", median)

# Sum
sum = np.sum(arr1)
print("Sum:", sum)

#4. Data Structures with Pandas: 
#a.Create a DataFrame from a CSV file or dictionary. 
#b.Perform basic data manipulation: filtering, sorting, and indexing.
# Create a DataFrame from a dictionary
import pandas as pd
data = {'Name': ['Aviral', 'Sid', 'Parul', 'Tanmay'],
    'Age': [19, 51, 48, 23],
    'Salary': [100000, 60000, 70000, 65000]
}
df = pd.DataFrame(data)
print("DataFrame:\n", df)

# Filter rows where Age is greater than 30
filter =df[df['Age']>22]
print("Filtered DataFrame:\n", filter)

# Sort the DataFrame by Salary in descending order
sorted_salary_df = df.sort_values(by='Salary', ascending=False)
print("Sorted DataFrame by Salary (Descending):\n", sorted_salary_df)

# Setting 'Name' as the index
df_index = df.set_index('Name')
print("DataFrame with 'Name' as Index:\n", df_index)

# 5. Basic Data Analysis: 
# Calculate descriptive statistics (mean, median, standard deviation) on a dataset 
# using Pandas and NumPy.
# Using pandas:
summary = df.describe()
print("Summary Statistics:\n", summary)
# Using numpy:

# Calculate the mean of the 'Age' column using NumPy
mean_age = np.mean(df['Age'])
print("Mean (Age) using NumPy:", mean_age)
# Calculate the median of the 'Salary' column using NumPy
median_salary = np.median(df['Salary'])
print("Median (Salary) using NumPy:", median_salary)
# Calculate the standard deviation of the 'Age' column using NumPy
std_dev_age = np.std(df['Age'])
print("Standard Deviation (Age) using NumPy:", std_dev_age)

# 6. Matplotlib Basics: 
#Plot a simple line chart. 
#Create a scatter plot. 
# LINE CHART:
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]  # X-axis values
y = [1, 4, 9, 16, 25]  # Y-axis values
plt.plot(x, y)
# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Chart')
# Show the plot
plt.show()

# SCATTER PLOT:
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]  # X-axis values
y = [1, 4, 9, 16, 25]  # Y-axis values
# Create a scatter plot
plt.scatter(x, y, color='red')
# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot Example')
# Show the plot
plt.show()

#7. Data Visualization with Seaborn: 
# Create a box plot to visualize data distributions. 
# Generate a histogram and a density plot. 
# BOX PLOT
import seaborn as sns
# Generating sample data
data1 = np.random.normal(loc=0, scale=1, size=100)
# Create a box plot
sns.boxplot(data=data1)
# Add a title and display the plot
plt.title("Box Plot Example")
plt.show()

#HISTOGRAM
import seaborn as sns
# Generating sample data (100 random values from a normal distribution)
data = np.random.normal(loc=0, scale=1, size=100)
# Create a histogram
sns.histplot(data, kde=False, bins=10)  # Setting kde=False to disable the density plot
# Add labels and title
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram Example")
# Show the plot
plt.show()

#DENSITY PLOT
import seaborn as sns
# Generating sample data (100 random values from a normal distribution)
data = np.random.normal(loc=0, scale=1, size=100)
# Create a density plot
sns.kdeplot(data, shade=True)
# Add labels and title
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Density Plot Example")
# Show the plot
plt.show()

# 8. Time Series Data: 
# Plot a time series using Pandas and Matplotlib.

import pandas as pd
import matplotlib.pyplot as plt
# Generate a time series of dates
dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
# Generate some random data for the time series
data = [150 + i + (i % 5) * 5 for i in range(30)]
# Create a DataFrame
df = pd.DataFrame({'Date': dates, 'Value': data})
df.set_index('Date', inplace=True)
# Plot the time series using Matplotlib
plt.figure(figsize=(10, 6))  # Set the size of the plot
plt.plot(df.index, df['Value'], marker='o', color='b', label='Value')
# Add title and labels
plt.title('Example Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45) #roation for better visualisation
plt.grid(True)
# Show the plot
plt.legend()
plt.tight_layout()
plt.show()

# 9. Correlation Analysis: 
# Generate a correlation heatmap using Seaborn. 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': [2, 3, 4, 5, 6],
    'D': [1, 3, 2, 5, 4]
}
df = pd.DataFrame(data)
print(df)
# Calculate the correlation matrix
corr_matrix = df.corr()
print(corr_matrix)
# Create the heatmap
plt.figure(figsize=(8, 6))  # Set the figure size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
# Add a title
plt.title('Correlation Heatmap')
# Show the plot
plt.show()

# 10. Data Aggregation:
# Group data using Pandas and visualize the results.
import pandas as pd

# Create a sample dataset
data = {
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [200, 150, 220, 180, 210, 160, 230, 190, 220, 170],
    'Quantity': [10, 15, 12, 20, 11, 16, 13, 18, 10, 14]
}
# this sample data is taken from the internet
df = pd.DataFrame(data)
print(df)

# Group data by 'Region' and 'Product' and calculate total sales and average quantity
grouped = df.groupby(['Region', 'Product']).agg(
    total_sales=('Sales', 'sum'),
    avg_quantity=('Quantity', 'mean')
).reset_index()
print(grouped)

# Plot a bar plot of total sales by Region and Product
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped, x='Region', y='total_sales', hue='Product')
# Add labels and title
plt.title('Total Sales by Region and Product', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
# Show the plot
plt.tight_layout()
plt.show()

# 11. Data Cleaning: 
# Handle missing values and perform data imputation.
import pandas as pd

# Load dataset
df = pd.read_csv('Task1/mtcars.csv')

# Check for missing values
missing_data = df.isnull().sum()
print(missing_data)

# Drop rows with any missing values
df_cleaned = df.dropna()
# Drop columns with any missing values
df_cleaned = df.dropna(axis=1)

#12. Combining Plots: 
# Create a figure with multiple subplots (line, bar, and scatter plots)
import matplotlib.pyplot as plt
import numpy as np
# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)
y2 = np.cos(x)
y3 = np.random.randn(100)

# Create a figure with 1 row and 3 columns of subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

# Line plot (first subplot)
axs[0].plot(x, y, label='Sine Wave', color='b')
axs[0].set_title('Line Plot')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].legend()

# Bar plot (second subplot)
axs[1].bar(x[:10], np.random.randint(1, 10, 10), color='g')
axs[1].set_title('Bar Plot')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')

# Scatter plot (third subplot)
axs[2].scatter(x, y3, color='r', label='Random Data')
axs[2].set_title('Scatter Plot')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
axs[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plot
plt.show()

# 13. Custom Visualization: 
# Experiment with customizing plot aesthetics (colors, labels, legends). 
import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)
# Create the plot
plt.plot(x, y, label='Sine Wave', color='blue', linestyle='-', linewidth=2)
# Add labels and title
plt.xlabel('X Axis', fontsize=12, color='purple')
plt.ylabel('Y Axis', fontsize=12, color='purple')
plt.title('Customized Plot', fontsize=14, color='darkred')
# Add a grid
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
# Add a legend
plt.legend(loc='upper right', fontsize=10)
# Show the plot
plt.tight_layout()
plt.show()

#14. Exploratory Data Analysis (EDA): 
# Conduct a brief EDA on a public dataset (e.g., Titanic, Iris).
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the Iris dataset from seaborn
df = sns.load_dataset('iris')
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Summary of the dataset (information about data types, non-null values)
print("Dataset Information:")
print(df.info())

# Descriptive statistics (summary of numerical features)
print("Descriptive Statistics:")
print(df.describe())

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())









