import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set page configuration
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("HR_Employee_Attrition.csv")
        return df
    except FileNotFoundError:
        st.error("Please ensure the HR_Employee_Attrition.csv file is in the same directory as this script.")
        st.stop()

# Function to preprocess data
@st.cache_data
def preprocess_data(df):
    # Convert categorical variables to numeric
    df_processed = df.copy()
    
    # Convert 'Attrition' to binary (0/1)
    df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})
    
    return df_processed

# Function to train model
@st.cache_resource
def train_model(df):
    # Define features and target
    y = df['Attrition']
    X = df.drop(['Attrition', 'EmployeeNumber'], axis=1, errors='ignore')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Create and train model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Get predictions and metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X_train, X_test, y_train, y_test, accuracy, numerical_cols, categorical_cols

# Function to get feature importance
@st.cache_data
def get_feature_importance(_model, _feature_names):
    # Extract the Random Forest classifier from the pipeline
    rf_classifier = _model.named_steps['classifier']
    
    # Get feature names after one-hot encoding
    categorical_cols = _model.named_steps['preprocessor'].transformers_[1][2]
    ohe = _model.named_steps['preprocessor'].transformers_[1][1]
    
    # Get feature importances and map to feature names
    importances = rf_classifier.feature_importances_
    
    # Get the names of all features after transformation
    all_feature_names = []
    all_feature_names.extend(_model.named_steps['preprocessor'].transformers_[0][2])  # Numerical columns
    
    # Get the categorical feature names after one-hot encoding
    cat_features = []
    for i, col in enumerate(categorical_cols):
        categories = ohe.categories_[i]
        for category in categories:
            cat_features.append(f"{col}_{category}")
    
    all_feature_names.extend(cat_features)
    
    # Map importance to features
    feature_importance = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

# Sidebar
def sidebar():
    st.sidebar.title("HR Analytics Dashboard")
    st.sidebar.markdown("### Navigation")
    
    page = st.sidebar.radio(
        "Select a Page",
        ["Overview", "Data Exploration", "Attrition Analysis", "Predictive Model", "Intervention Simulator"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard analyzes employee attrition data to identify risk factors "
        "and simulate potential HR interventions to improve retention."
    )
    
    return page

# Overview page
def overview_page(df):
    st.title("HR Analytics Dashboard")
    st.markdown("## Employee Attrition Analysis and Prediction")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Total employees
    col1.metric("Total Employees", df.shape[0])
    
    # Attrition rate
    attrition_rate = round(df[df['Attrition'] == 1].shape[0] / df.shape[0] * 100, 2)
    col2.metric("Attrition Rate", f"{attrition_rate}%")
    
    # Average age
    col3.metric("Average Age", round(df['Age'].mean(), 1))
    
    # Average tenure
    col4.metric("Average Tenure", round(df['YearsAtCompany'].mean(), 1))
    
    # Department breakdown
    st.markdown("### Department Distribution")
    dept_count = df['Department'].value_counts().reset_index()
    dept_count.columns = ['Department', 'Count']
    
    fig = px.pie(dept_count, values='Count', names='Department', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)
    
    # Attrition by department
    st.markdown("### Attrition by Department")
    dept_attrition = df.groupby('Department')['Attrition'].mean().reset_index()
    dept_attrition['Attrition'] = dept_attrition['Attrition'] * 100
    
    fig = px.bar(dept_attrition, x='Department', y='Attrition', 
                 title='Attrition Rate (%) by Department',
                 labels={'Attrition': 'Attrition Rate (%)'})
    st.plotly_chart(fig, use_container_width=True)

# Data exploration page
def data_exploration_page(df):
    st.title("Data Exploration")
    
    # Show data sample
    st.markdown("### Data Sample")
    st.dataframe(df.head())
    
    # Data summary
    st.markdown("### Data Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Dataset Features:", df.columns.tolist())
    
    with col2:
        missing_values = df.isnull().sum().sum()
        st.write("Missing Values:", missing_values)
        st.write("Attrition Count:", df['Attrition'].value_counts().to_dict())
    
    # Correlation analysis
    st.markdown("### Correlation with Attrition")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation with Attrition
    attrition_corr = numeric_df.corr()['Attrition'].drop('Attrition')
    attrition_corr = attrition_corr.sort_values(key=abs, ascending=False)
    
    # Create a DataFrame for display with better formatting
    attrition_corr_df = pd.DataFrame({
        'Feature': attrition_corr.index,
        'Correlation': attrition_corr.values
    })
    
    # Show the correlation values in a horizontal bar chart
    fig = px.bar(
        attrition_corr_df.head(15), 
        x='Correlation', 
        y='Feature',
        orientation='h',
        color='Correlation',
        color_continuous_scale=px.colors.diverging.RdBu_r,
        title='Top 15 Features Correlated with Attrition',
        range_color=[-1, 1]
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Option to see the full correlation matrix
    if st.checkbox("Show full correlation matrix"):
        st.markdown("### Full Correlation Matrix")
        
        # Calculate correlation matrix
        corr = numeric_df.corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create a mask for the upper triangle to avoid redundancy
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Plot the heatmap with the mask
        sns.heatmap(
            corr, 
            mask=mask,
            annot=False,  # Too many values to show annotations clearly
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add a slider to filter correlations by strength
        corr_threshold = st.slider(
            "Show correlations stronger than:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Find and display strong correlations
        strong_corrs = []
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) >= corr_threshold:
                    strong_corrs.append({
                        'Feature 1': corr.columns[i],
                        'Feature 2': corr.columns[j],
                        'Correlation': corr.iloc[i, j]
                    })
        
        if strong_corrs:
            strong_corrs_df = pd.DataFrame(strong_corrs).sort_values('Correlation', key=abs, ascending=False)
            st.markdown(f"### Strong Correlations (|r| ≥ {corr_threshold})")
            st.dataframe(strong_corrs_df, use_container_width=True)
        else:
            st.info(f"No correlations stronger than {corr_threshold} found.")
    
    # Feature distributions
    st.markdown("### Feature Distributions")
    
    # Let user select features to visualize
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_feature = st.selectbox("Select a feature to visualize", numeric_cols)
    
    # Create histogram with density plot
    fig = px.histogram(df, x=selected_feature, color='Attrition', 
                       marginal="box", 
                       labels={'Attrition': 'Left Company'},
                       color_discrete_map={0: 'blue', 1: 'red'},
                       title=f'Distribution of {selected_feature} by Attrition')
    
    st.plotly_chart(fig, use_container_width=True)

# Attrition analysis page
def attrition_analysis_page(df):
    st.title("Attrition Analysis")
    
    # Attrition by age group
    st.markdown("### Attrition by Age Group")
    
    # Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], 
                            labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
    
    age_attrition = df.groupby('AgeGroup')['Attrition'].mean().reset_index()
    age_attrition['Attrition'] = age_attrition['Attrition'] * 100
    
    fig = px.bar(age_attrition, x='AgeGroup', y='Attrition',
                 labels={'Attrition': 'Attrition Rate (%)', 'AgeGroup': 'Age Group'},
                 title='Attrition Rate by Age Group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Attrition by job satisfaction
    st.markdown("### Attrition by Job Satisfaction")
    
    job_sat_attrition = df.groupby('JobSatisfaction')['Attrition'].mean().reset_index()
    job_sat_attrition['Attrition'] = job_sat_attrition['Attrition'] * 100
    
    fig = px.line(job_sat_attrition, x='JobSatisfaction', y='Attrition', markers=True,
                  labels={'Attrition': 'Attrition Rate (%)', 'JobSatisfaction': 'Job Satisfaction Level'},
                  title='Attrition Rate by Job Satisfaction')
    st.plotly_chart(fig, use_container_width=True)
    
    # Attrition by overtime
    st.markdown("### Attrition by Overtime")
    
    overtime_attrition = df.groupby('OverTime')['Attrition'].mean().reset_index()
    overtime_attrition['Attrition'] = overtime_attrition['Attrition'] * 100
    
    fig = px.bar(overtime_attrition, x='OverTime', y='Attrition',
                 labels={'Attrition': 'Attrition Rate (%)', 'OverTime': 'Works Overtime'},
                 title='Attrition Rate by Overtime')
    st.plotly_chart(fig, use_container_width=True)
    
    # Attrition by salary
    st.markdown("### Attrition by Monthly Income")
    
    # Create income buckets
    df['IncomeBucket'] = pd.qcut(df['MonthlyIncome'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    income_attrition = df.groupby('IncomeBucket')['Attrition'].mean().reset_index()
    income_attrition['Attrition'] = income_attrition['Attrition'] * 100
    
    fig = px.bar(income_attrition, x='IncomeBucket', y='Attrition',
                 labels={'Attrition': 'Attrition Rate (%)', 'IncomeBucket': 'Income Level'},
                 title='Attrition Rate by Income Level')
    st.plotly_chart(fig, use_container_width=True)

# Predictive model page
def predictive_model_page(df, model, X_test, y_test, accuracy, feature_importance):
    st.title("Predictive Model")
    
    # Model performance
    st.markdown("### Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display model accuracy
        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
        
        # Display confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        # Show classification report
        st.text("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    
    # Feature importance
    st.markdown("### Feature Importance")
    
    # Get top 15 features
    top_features = feature_importance.head(15)
    
    fig = px.bar(top_features, x='Importance', y='Feature', orientation='h',
                 title='Top 15 Features for Attrition Prediction',
                 labels={'Importance': 'Feature Importance', 'Feature': 'Feature Name'})
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Employee risk prediction
    st.markdown("### Employee Risk Assessment")
    st.info("Use this section to identify employees at high risk of attrition.")
    
    # For demo purposes, let's create a risk score for each employee in the dataset
    df_risk = df.copy()
    
    # Predict probability of attrition for each employee
    X = df_risk.drop(['Attrition', 'EmployeeNumber'], axis=1, errors='ignore')
    df_risk['AttritionRisk'] = model.predict_proba(X)[:, 1]
    
    # Display high-risk employees (top 10%)
    risk_threshold = df_risk['AttritionRisk'].quantile(0.9)
    high_risk = df_risk[df_risk['AttritionRisk'] >= risk_threshold].sort_values('AttritionRisk', ascending=False)
    
    # Select relevant columns for display
    display_cols = ['EmployeeNumber', 'Department', 'JobRole', 'Age', 'MonthlyIncome', 
                   'OverTime', 'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany', 'AttritionRisk']
    display_cols = [col for col in display_cols if col in high_risk.columns]
    
    # Display high-risk employees
    st.markdown("#### High-Risk Employees")
    st.dataframe(high_risk[display_cols].head(10))
    
    # Risk distribution
    st.markdown("#### Risk Score Distribution")
    
    fig = px.histogram(df_risk, x='AttritionRisk', nbins=50,
                     title='Distribution of Attrition Risk Scores',
                     labels={'AttritionRisk': 'Attrition Risk Score'})
    st.plotly_chart(fig, use_container_width=True)

# Intervention simulator page
def intervention_simulator_page(df, model, numerical_cols, categorical_cols):
    st.title("Intervention Simulator")
    st.markdown("Simulate the impact of HR interventions on employee retention")
    
    # Select an employee to analyze
    st.markdown("### Select Employee to Analyze")
    employee_id = st.selectbox("Select Employee ID", df['EmployeeNumber'].unique())
    
    # Get employee data
    employee_data = df[df['EmployeeNumber'] == employee_id].copy()
    employee_row = employee_data.iloc[0]
    
    # Display employee info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Employee Details")
        st.write(f"**Department:** {employee_row['Department']}")
        st.write(f"**Job Role:** {employee_row['JobRole']}")
        st.write(f"**Age:** {employee_row['Age']}")
        st.write(f"**Gender:** {employee_row['Gender']}")
        
    with col2:
        st.markdown("#### Work Details")
        st.write(f"**Years at Company:** {employee_row['YearsAtCompany']}")
        st.write(f"**Monthly Income:** ${employee_row['MonthlyIncome']:,.2f}")
        st.write(f"**Overtime:** {employee_row['OverTime']}")
        st.write(f"**Job Level:** {employee_row['JobLevel']}")
        
    with col3:
        st.markdown("#### Satisfaction Metrics")
        st.write(f"**Job Satisfaction:** {employee_row['JobSatisfaction']}/4")
        st.write(f"**Environment Satisfaction:** {employee_row['EnvironmentSatisfaction']}/4")
        st.write(f"**Work-Life Balance:** {employee_row['WorkLifeBalance']}/4")
        st.write(f"**Relationship Satisfaction:** {employee_row['RelationshipSatisfaction']}/4")
    
    try:
        # Calculate base risk
        X_pred = employee_data.drop(['Attrition'], axis=1)
        if 'EmployeeNumber' in X_pred.columns:
            X_pred = X_pred.drop(['EmployeeNumber'], axis=1)
        
        # Calculate base risk
        base_risk = model.predict_proba(X_pred)[0][1]
        
        # Display risk score
        risk_color = "red" if base_risk > 0.7 else "orange" if base_risk > 0.4 else "green"
        
        st.markdown(f"### Current Attrition Risk: <span style='color:{risk_color};'>{base_risk*100:.2f}%</span>", unsafe_allow_html=True)
        
        # Intervention simulation
        st.markdown("### Simulate Interventions")
        st.info("Adjust these factors to see how they might affect the employee's attrition risk.")
        
        # Create a copy of the employee data for simulation
        employee_data_sim = employee_data.copy()
        
        # Define potential interventions
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary adjustment
            salary_increase = st.slider("Salary Adjustment (%)", -10, 30, 0, 5)
            new_salary = employee_row['MonthlyIncome'] * (1 + salary_increase/100)
            
            # Job satisfaction improvement
            if employee_row['JobSatisfaction'] < 4:
                job_sat_improvement = st.slider("Job Satisfaction Improvement", 
                                             0, 4 - employee_row['JobSatisfaction'], 0)
            else:
                job_sat_improvement = 0
            
            # Work-life balance improvement
            if employee_row['WorkLifeBalance'] < 4:
                wl_improvement = st.slider("Work-Life Balance Improvement", 
                                        0, 4 - employee_row['WorkLifeBalance'], 0)
            else:
                wl_improvement = 0
        
        with col2:
            # Overtime change
            reduce_overtime = False
            if employee_row['OverTime'] == 'Yes':
                reduce_overtime = st.checkbox("Reduce Overtime")
            
            # Job level promotion
            promote = False
            if employee_row['JobLevel'] < 5:
                promote = st.checkbox("Promote to Next Job Level")
            
            # Environment satisfaction improvement
            if employee_row['EnvironmentSatisfaction'] < 4:
                env_improvement = st.slider("Environment Satisfaction Improvement", 
                                         0, 4 - employee_row['EnvironmentSatisfaction'], 0)
            else:
                env_improvement = 0
        
        if st.button("Calculate New Risk"):
            # Apply interventions to simulation data
            employee_data_sim['MonthlyIncome'] = new_salary
            
            if job_sat_improvement > 0:
                employee_data_sim['JobSatisfaction'] = employee_row['JobSatisfaction'] + job_sat_improvement
            
            if wl_improvement > 0:
                employee_data_sim['WorkLifeBalance'] = employee_row['WorkLifeBalance'] + wl_improvement
            
            if reduce_overtime:
                employee_data_sim['OverTime'] = 'No'
            
            if promote:
                employee_data_sim['JobLevel'] = employee_row['JobLevel'] + 1
            
            if env_improvement > 0:
                employee_data_sim['EnvironmentSatisfaction'] = employee_row['EnvironmentSatisfaction'] + env_improvement
            
            # Prepare data for prediction
            X_sim = employee_data_sim.drop(['Attrition'], axis=1)
            if 'EmployeeNumber' in X_sim.columns:
                X_sim = X_sim.drop(['EmployeeNumber'], axis=1)
            
            # Calculate new risk with interventions
            new_risk = model.predict_proba(X_sim)[0][1]
            
            # Display new risk score
            new_risk_color = "red" if new_risk > 0.7 else "orange" if new_risk > 0.4 else "green"
            
            # Calculate risk reduction
            risk_reduction = base_risk - new_risk
            risk_reduction_pct = (risk_reduction / base_risk) * 100 if base_risk > 0 else 0
            
            st.markdown("### Intervention Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**New Attrition Risk:** <span style='color:{new_risk_color};'>{new_risk*100:.2f}%</span>", unsafe_allow_html=True)
            
            with col2:
                if risk_reduction > 0:
                    st.markdown(f"**Risk Reduction:** <span style='color:green;'>{risk_reduction_pct:.2f}%</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Risk Change:** <span style='color:red;'>{risk_reduction_pct:.2f}%</span>", unsafe_allow_html=True)
            
            # Estimate intervention cost
            st.markdown("### Intervention Cost Analysis")
            
            # Calculate cost of salary increase
            annual_salary_increase = (new_salary - employee_row['MonthlyIncome']) * 12
            
            # Estimate cost of turnover (typically 1-2x annual salary)
            annual_salary = employee_row['MonthlyIncome'] * 12
            turnover_cost = annual_salary * 1.5  # Using 1.5x annual salary as turnover cost
            
            # Calculate ROI
            intervention_cost = annual_salary_increase  # Simplified, only including salary increase
            risk_reduction_value = turnover_cost * risk_reduction  # Expected value of risk reduction
            roi = (risk_reduction_value - intervention_cost) / intervention_cost * 100 if intervention_cost > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Annual Salary Increase", f"${annual_salary_increase:,.2f}")
            
            with col2:
                st.metric("Estimated Turnover Cost", f"${turnover_cost:,.2f}")
            
            with col3:
                if intervention_cost > 0:
                    st.metric("ROI of Intervention", f"{roi:.2f}%")
                else:
                    st.metric("ROI of Intervention", "N/A")
            
            # Recommendations
            st.markdown("### Recommended Actions")
            
            # Generate recommendations based on employee data and feature importance
            recommendations = []
            
            if employee_row['OverTime'] == 'Yes' and not reduce_overtime:
                recommendations.append("Consider reducing overtime requirements for this employee.")
            
            if employee_row['JobSatisfaction'] < 3:
                recommendations.append("Schedule a career development discussion to improve job satisfaction.")
            
            if employee_row['WorkLifeBalance'] < 3:
                recommendations.append("Explore flexible work arrangements to improve work-life balance.")
            
            if 'YearsSinceLastPromotion' in df.columns and employee_row['YearsSinceLastPromotion'] > 2 and employee_row['JobLevel'] < 5:
                recommendations.append("Evaluate for potential promotion or expanded responsibilities.")
                
            if employee_row['MonthlyIncome'] < df[df['JobRole'] == employee_row['JobRole']]['MonthlyIncome'].median():
                recommendations.append("Consider a compensation review as employee is below median for their role.")
            
            if not recommendations:
                recommendations.append("No specific interventions required at this time. Continue regular engagement.")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
                
    except Exception as e:
        st.error(f"Error in risk calculation: {str(e)}")
        st.info("This might be due to mismatches in the data structure. Try selecting a different employee.")

# Main function
def main():
    page = sidebar()
    
    # Load and preprocess data
    with st.spinner('Loading data...'):
        df = load_data()
        df_processed = preprocess_data(df)
    
    # Train model
    with st.spinner('Training model...'):
        model, X_train, X_test, y_train, y_test, accuracy, numerical_cols, categorical_cols = train_model(df_processed)
    
    try:
        # Get feature importance
        feature_importance = get_feature_importance(model, df_processed.columns.drop('Attrition'))
    except Exception as e:
        st.warning(f"Couldn't calculate feature importance: {str(e)}")
        feature_importance = pd.DataFrame({
            'Feature': df_processed.columns.drop('Attrition'),
            'Importance': np.random.rand(len(df_processed.columns) - 1)
        }).sort_values('Importance', ascending=False)
    
    # Display selected page
    if page == "Overview":
        overview_page(df_processed)
    elif page == "Data Exploration":
        data_exploration_page(df_processed)
    elif page == "Attrition Analysis":
        attrition_analysis_page(df_processed)
    elif page == "Predictive Model":
        predictive_model_page(df_processed, model, X_test, y_test, accuracy, feature_importance)
    elif page == "Intervention Simulator":
        intervention_simulator_page(df_processed, model, numerical_cols, categorical_cols)

if __name__ == "__main__":
    main()