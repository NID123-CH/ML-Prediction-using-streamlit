import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd
import matplotlib as plt
from model_code import load_model, predict_life_expectancy, get_feature_importance, train_model

# Set page layout to wide
st.set_page_config(layout="wide")

# Header
st.title("Worldwide Analysis of Quality of Life and Economic Factors")

# Subtitle
st.markdown("""
### This app enables you to explore the relationships between poverty, 
life expectancy, and GDP across various countries and years. 
Use the panels to select options and interact with the data.
""")

# Tabs
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])

# Content for each tab
with tab1:
    st.header("Global Overview")
    st.write("Here you can explore a global overview of quality of life and economic factors.")
    

with tab2:
    st.header("Country Deep Dive")
    st.write("Dive into specific countries and analyze their economic and quality of life trends.")

with tab3:
    st.header("Data Explorer")
    st.write("Explore and interact with the raw data, filter, and analyze trends over time.") 
    
    
# Read the dataset
    def load_data():
        data = pd.read_csv("global_development_data.csv")
        return data
    
    data = load_data()

    # Show dataset in the tab
    st.write("### Global Development Data")
    st.write("The dataset contains information on poverty, life expectancy, and GDP across multiple countries and years.")

    # Multiselect for countries
    countries = data['country'].unique()  # Assuming the dataset has a 'Country' column
    selected_countries = st.multiselect("Select Countries", countries, default=countries)

    # Slider for year range
    min_year = int(data['year'].min())  # Assuming the dataset has a 'Year' column
    max_year = int(data['year'].max())
    year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))

    # Filter dataset based on selection
    filtered_data = data[
        (data['country'].isin(selected_countries)) &
        (data['year'] >= year_range[0]) &
        (data['year'] <= year_range[1])
    ]

    # Display the filtered data
    st.write("### Filtered Data")
    st.dataframe(filtered_data)

    # Button to download the filtered data
    
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(filtered_data)

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_global_development_data.csv',
        mime='text/csv',
    ) 
    
with tab1:
    
    # Slider for year selection
    selected_year = st.slider("Select Year", int(data['year'].min()), int(data['year'].max()), int(data['year'].min()))
    
    # Filter data based on selected year
    year_data = data[data['year'] == selected_year]

    # Calculate metrics
    mean_life_expectancy = year_data['Life Expectancy (IHME)'].mean()
    median_gdp_per_capita = year_data['GDP per capita'].median()
    mean_poverty_rate = year_data['headcount_ratio_upper_mid_income_povline'].mean()
    num_countries = year_data['country'].nunique()

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Mean Life Expectancy", value=f"{mean_life_expectancy:.2f} years")
        st.write("The average life expectancy across selected countries.")

    with col2:
        st.metric(label="Median GDP per Capita", value=f"${median_gdp_per_capita:,.2f}")
        st.write("The median GDP per capita for the selected year.")

    with col3:
        st.metric(label="Mean Poverty Rate (Upper Middle Income)", value=f"{mean_poverty_rate:.2f}%")
        st.write("The average headcount ratio for upper middle-income poverty line.")

    with col4:
        st.metric(label="Number of Countries", value=num_countries)
        st.write("The number of countries in the dataset for the selected year.")
        
     # Create scatterplot using Plotly
    scatter_plot = px.scatter(
        year_data,
        x="GDP per capita",
        y="Life Expectancy (IHME)",
        size="Population",  # Size by GDP per capita
        color="country",  # Color by Life Expectancy
        hover_name="country",  # Hover info to show country names
        log_x=True,  # Logarithmic scale for GDP per capita
        title="GDP per Capita vs Life Expectancy",
        labels={
            "GDP per capita": "GDP per Capita (USD)",
            "Life Expectancy (IHME)": "Life Expectancy (years)"
        }
    )

    # Show plotly chart in Streamlit
    st.plotly_chart(scatter_plot, use_container_width=True)
    
    # Load or train the model
if 'model' not in st.session_state:
    train_model()
    st.session_state.model = load_model()

# Streamlit app layout
st.title('Life Expectancy Prediction')



# Get necessary statistics for input fields
# Example statistics (replace with your actual calculations)
median_gdp = 10000  # Placeholder for median GDP
headcount_ratio_upper_mid_income_povline_mean = 20  # Placeholder for mean headcount ratio
min_year = 2000  # Placeholder for minimum year in your dataset
max_year = 2025  # Placeholder for maximum year in your dataset


# Input fields for the user
st.write("Predict Life Expectancy:")
input_gdp = st.number_input("Enter GDP per capita", min_value=0, value=median_gdp)
input_poverty = st.number_input("Enter headcount ratio upper mid income povline", min_value=0, value=headcount_ratio_upper_mid_income_povline_mean)
input_year = st.number_input("Enter Year for prediction", min_value=min_year, max_value=max_year, value=2000)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'GDP per capita': [input_gdp],
    'headcount_ratio_upper_mid_income_povline': [input_poverty],
    'year': [input_year]
})

# Make predictions
model = st.session_state.model
if st.button('Predict Life Expectancy'):
    prediction = model.predict(input_data)
    st.write(f"Estimated Life Expectancy: {prediction[0]:.2f} years")

# Feature importance visualization
if st.button('Show Feature Importance'):
    features = ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']
    feature_importances = model.feature_importances_
    
    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Create a bar plot for feature importance using Plotly
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
    st.plotly_chart(fig)

    
    