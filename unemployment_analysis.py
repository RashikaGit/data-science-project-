# ============================================================================
# TASK 2: UNEMPLOYMENT ANALYSIS WITH PYTHON
# ============================================================================

print("="*70)
print(" " * 20 + "UNEMPLOYMENT ANALYSIS")
print("="*70)

# Step 1: Import Libraries
print("\n[STEP 1] Importing libraries...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
print("✓ Libraries imported successfully!")

# Step 2: Load Dataset
print("\n[STEP 2] Loading dataset...")

# Try to load actual dataset, otherwise create sample data
try:
    # Agar dataset file hai toh load karein
    df = pd.read_csv('unemployment_data.csv')
    print("✓ Dataset loaded from file!")
except:
    # Sample data create karein (2015-2022 tak)
    print("Creating sample unemployment dataset...")
    
    np.random.seed(42)
    
    # Countries aur regions
    countries = ['United States', 'United Kingdom', 'Germany', 'France', 
                 'India', 'China', 'Japan', 'Brazil', 'Canada', 'Australia']
    regions = ['North America', 'Europe', 'Asia', 'South America', 'Oceania']
    
    # Date range (Monthly data 2015-2022)
    dates = pd.date_range(start='2015-01-01', end='2022-12-01', freq='MS')
    
    data = []
    
    for country in countries:
        # Base unemployment rate har country ke liye alag
        if country in ['United States', 'United Kingdom', 'France']:
            base_rate = np.random.uniform(4, 6)
        elif country in ['Germany', 'Japan', 'China']:
            base_rate = np.random.uniform(3, 5)
        elif country in ['India', 'Brazil']:
            base_rate = np.random.uniform(6, 9)
        else:
            base_rate = np.random.uniform(4, 7)
        
        for date in dates:
            # Seasonal variation
            month = date.month
            seasonal = np.sin(2 * np.pi * month / 12) * 0.5
            
            # COVID-19 impact (March 2020 - Dec 2021)
            if date >= datetime(2020, 3, 1) and date <= datetime(2021, 12, 1):
                covid_impact = np.random.uniform(3, 8)
                if date <= datetime(2020, 6, 1):  # Peak COVID
                    covid_impact = np.random.uniform(5, 10)
            else:
                covid_impact = 0
            
            # Trend (thoda gradual decrease)
            years_from_start = (date.year - 2015) + (date.month / 12)
            trend = -0.1 * years_from_start
            
            # Random noise
            noise = np.random.normal(0, 0.3)
            
            # Final unemployment rate
            unemployment_rate = base_rate + seasonal + covid_impact + trend + noise
            unemployment_rate = max(2, min(20, unemployment_rate))  # Between 2-20%
            
            # Labor force participation rate
            labor_force = np.random.uniform(60, 75)
            
            # Employed population
            employed = labor_force * (100 - unemployment_rate) / 100
            
            data.append({
                'Date': date,
                'Country': country,
                'Region': regions[countries.index(country) % len(regions)],
                'Unemployment_Rate': round(unemployment_rate, 2),
                'Labor_Force_Participation': round(labor_force, 2),
                'Employed_Population': round(employed, 2)
            })
    
    df = pd.DataFrame(data)
    print(f"✓ Sample dataset created: {df.shape[0]} records")

# Display basic info
print(f"\nDataset Shape: {df.shape}")
print(f"\nColumns: {', '.join(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")

# Step 3: Data Cleaning
print("\n" + "="*70)
print("[STEP 3] Data Cleaning & Preprocessing")
print("="*70)

# Check for missing values
print(f"\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

# Remove duplicates if any
duplicates = df.duplicated().sum()
if duplicates > 0:
    print(f"Removing {duplicates} duplicate records...")
    df = df.drop_duplicates()

# Convert Date column to datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')

print("✓ Data cleaning completed!")

# Step 4: Exploratory Data Analysis
print("\n" + "="*70)
print("[STEP 4] Exploratory Data Analysis")
print("="*70)

print(f"\nStatistical Summary:")
print(df.describe())

print(f"\nCountries in dataset: {df['Country'].nunique()}")
print(f"Time period: {df['Date'].min()} to {df['Date'].max()}")

# Step 5: Data Visualization
print("\n" + "="*70)
print("[STEP 5] Creating Visualizations...")
print("="*70)

# 5.1 Overall Unemployment Trend Over Time
print("\nCreating: Overall Unemployment Trend...")
plt.figure(figsize=(14, 7))
if 'Country' in df.columns:
    for country in df['Country'].unique()[:5]:  # Top 5 countries
        country_data = df[df['Country'] == country].sort_values('Date')
        plt.plot(country_data['Date'], country_data['Unemployment_Rate'], 
                label=country, linewidth=2, alpha=0.8)
    
    # COVID period highlight
    plt.axvspan(datetime(2020, 3, 1), datetime(2021, 12, 1), 
                alpha=0.3, color='red', label='COVID-19 Period')
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    plt.title('Unemployment Rate Trend Over Time (2015-2022)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 5.2 COVID-19 Impact Analysis
print("\nCreating: COVID-19 Impact Analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Before COVID (2019)
before_covid = df[df['Year'] == 2019]
axes[0, 0].hist(before_covid['Unemployment_Rate'], bins=20, 
                color='green', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(before_covid['Unemployment_Rate'].mean(), 
                   color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {before_covid['Unemployment_Rate'].mean():.2f}%")
axes[0, 0].set_title('Before COVID-19 (2019)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Unemployment Rate (%)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# During COVID (2020-2021)
during_covid = df[(df['Year'] >= 2020) & (df['Year'] <= 2021)]
axes[0, 1].hist(during_covid['Unemployment_Rate'], bins=20, 
                color='red', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(during_covid['Unemployment_Rate'].mean(), 
                   color='darkred', linestyle='--', linewidth=2,
                   label=f"Mean: {during_covid['Unemployment_Rate'].mean():.2f}%")
axes[0, 1].set_title('During COVID-19 (2020-2021)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Unemployment Rate (%)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# After COVID (2022)
after_covid = df[df['Year'] == 2022]
axes[1, 0].hist(after_covid['Unemployment_Rate'], bins=20, 
                color='blue', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(after_covid['Unemployment_Rate'].mean(), 
                   color='darkblue', linestyle='--', linewidth=2,
                   label=f"Mean: {after_covid['Unemployment_Rate'].mean():.2f}%")
axes[1, 0].set_title('After COVID-19 (2022)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Unemployment Rate (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Comparison
periods = ['Before COVID\n(2019)', 'During COVID\n(2020-2021)', 'After COVID\n(2022)']
means = [before_covid['Unemployment_Rate'].mean(), 
         during_covid['Unemployment_Rate'].mean(),
         after_covid['Unemployment_Rate'].mean()]
colors = ['green', 'red', 'blue']
axes[1, 1].bar(periods, means, color=colors, edgecolor='black', linewidth=1.2)
axes[1, 1].set_title('Average Unemployment: Period Comparison', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Average Unemployment Rate (%)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(means):
    axes[1, 1].text(i, v + 0.2, f'{v:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# 5.3 Country-wise Comparison
print("\nCreating: Country-wise Unemployment Comparison...")
if 'Country' in df.columns:
    plt.figure(figsize=(12, 8))
    country_avg = df.groupby('Country')['Unemployment_Rate'].mean().sort_values(ascending=False)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(country_avg)))
    bars = plt.bar(country_avg.index, country_avg.values, color=colors, 
                   edgecolor='black', linewidth=1.2)
    
    plt.title('Average Unemployment Rate by Country (2015-2022)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Country', fontsize=12, fontweight='bold')
    plt.ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, country_avg.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# 5.4 Monthly Seasonal Patterns
print("\nCreating: Monthly Seasonal Patterns...")
if 'Month_Name' in df.columns:
    plt.figure(figsize=(10, 6))
    monthly_avg = df.groupby('Month_Name')['Unemployment_Rate'].mean()
    
    # Month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_avg = monthly_avg.reindex(month_order)
    
    sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker='o', 
                 linewidth=2.5, markersize=8, color='purple')
    
    plt.title('Monthly Seasonal Pattern in Unemployment', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontsize=12, fontweight='bold')
    plt.ylabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 5.5 Regional Analysis
print("\nCreating: Regional Analysis...")
if 'Region' in df.columns:
    plt.figure(figsize=(10, 6))
    regional_data = df.groupby('Region')['Unemployment_Rate'].agg(['mean', 'std']).sort_values('mean', ascending=False)
    
    plt.barh(regional_data.index, regional_data['mean'], 
             xerr=regional_data['std'], color='coral', alpha=0.7, edgecolor='black')
    
    plt.title('Regional Unemployment Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Average Unemployment Rate (%)', fontsize=12, fontweight='bold')
    plt.ylabel('Region', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# 5.6 Year-wise Trend
print("\nCreating: Year-wise Trend...")
if 'Year' in df.columns:
    yearly_stats = df.groupby('Year')['Unemployment_Rate'].agg(['mean', 'median', 'std'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_stats.index, yearly_stats['mean'], marker='o', 
             linewidth=3, markersize=10, label='Mean', color='blue')
    plt.plot(yearly_stats.index, yearly_stats['median'], marker='s', 
             linewidth=3, markersize=10, label='Median', color='green')
    plt.fill_between(yearly_stats.index, 
                     yearly_stats['mean'] - yearly_stats['std'],
                     yearly_stats['mean'] + yearly_stats['std'],
                     alpha=0.3, color='blue', label='±1 Std Dev')
    
    plt.title('Year-wise Unemployment Trend', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Unemployment Rate (%)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(yearly_stats.index)
    plt.tight_layout()
    plt.show()

# Step 6: COVID-19 Impact Analysis
print("\n" + "="*70)
print("[STEP 6] COVID-19 Impact Analysis")
print("="*70)

# Calculate COVID impact metrics
pre_covid = df[df['Year'] == 2019]['Unemployment_Rate']
covid_peak = df[(df['Year'] == 2020) & (df['Month'].isin([4, 5, 6, 7, 8]))]['Unemployment_Rate']
post_covid = df[df['Year'] == 2022]['Unemployment_Rate']

print(f"""
COVID-19 IMPACT METRICS:

📊 Average Unemployment Rates:
   • Pre-COVID (2019):     {pre_covid.mean():.2f}%
   • COVID Peak (2020):    {covid_peak.mean():.2f}%
   • Post-COVID (2022):    {post_covid.mean():.2f}%

📈 Change Analysis:
   • Increase during COVID: {covid_peak.mean() - pre_covid.mean():.2f} percentage points
   • Recovery by 2022:      {covid_peak.mean() - post_covid.mean():.2f} percentage points
   • Remaining gap:         {post_covid.mean() - pre_covid.mean():.2f} percentage points

📉 Percentage Change:
   • Peak increase:         {((covid_peak.mean() / pre_covid.mean()) - 1) * 100:.1f}%
   • Current vs Pre-COVID:  {((post_covid.mean() / pre_covid.mean()) - 1) * 100:.1f}%
""")

# Step 7: Key Patterns & Trends
print("\n" + "="*70)
print("[STEP 7] Key Patterns & Trends Identified")
print("="*70)

# Find highest and lowest unemployment periods
highest_month = df.loc[df['Unemployment_Rate'].idxmax()]
lowest_month = df.loc[df['Unemployment_Rate'].idxmin()]

print(f"""
KEY FINDINGS:

📅 Time Period Analyzed: {df['Date'].min().strftime('%B %Y')} to {df['Date'].max().strftime('%B %Y')}
📊 Total Records: {df.shape[0]}
🌍 Countries/Regions: {df['Country'].nunique() if 'Country' in df.columns else 'N/A'}

📈 UNEMPLOYMENT STATISTICS:
   • Overall Mean:    {df['Unemployment_Rate'].mean():.2f}%
   • Median:          {df['Unemployment_Rate'].median():.2f}%
   • Standard Dev:    {df['Unemployment_Rate'].std():.2f}%
   • Minimum:         {df['Unemployment_Rate'].min():.2f}%
   • Maximum:         {df['Unemployment_Rate'].max():.2f}%

 HIGHEST UNEMPLOYMENT:
   • {highest_month['Country'] if 'Country' in df.columns else 'N/A'} 
   • Date: {highest_month['Date'].strftime('%B %Y')}
   • Rate: {highest_month['Unemployment_Rate']:.2f}%

🔻 LOWEST UNEMPLOYMENT:
   • {lowest_month['Country'] if 'Country' in df.columns else 'N/A'}
   • Date: {lowest_month['Date'].strftime('%B %Y')}
   • Rate: {lowest_month['Unemployment_Rate']:.2f}%
""")

# Step 8: Policy Insights
print("\n" + "="*70)
print("[STEP 8] Economic & Social Policy Insights")
print("="*70)

print(f"""
💡 POLICY RECOMMENDATIONS:

1. COVID-19 RECOVERY STRATEGIES:
   • Unemployment increased by {covid_peak.mean() - pre_covid.mean():.2f}% during COVID peak
   • Focus on job creation in hardest-hit sectors
   • Implement skills development programs for displaced workers

2. SEASONAL EMPLOYMENT PLANNING:
   • Identify months with historically high unemployment
   • Create temporary employment programs during lean periods
   • Encourage seasonal industries to plan ahead

3. REGIONAL DEVELOPMENT:
   • Target high-unemployment regions with economic incentives
   • Invest in infrastructure projects in affected areas
   • Promote remote work opportunities

4. LONG-TERM STRATEGIES:
   • Current unemployment is {((post_covid.mean() / pre_covid.mean()) - 1) * 100:.1f}% vs pre-COVID levels
   • Focus on economic diversification
   • Strengthen social safety nets
   • Invest in education and retraining programs

5. EARLY WARNING SYSTEMS:
   • Monitor unemployment trends monthly
   • Identify at-risk sectors early
   • Prepare contingency plans for economic shocks

6. YOUTH & VULNERABLE GROUPS:
   • Special employment programs for new entrants
   • Support for small businesses and startups
   • Enhanced unemployment benefits during crises
""")

# Final Summary
print("\n" + "="*70)
print(" " * 25 + "PROJECT COMPLETED!")
print("="*70)
print(f"""
✅ All analysis tasks completed successfully!

📊 DATASET SUMMARY:
   • Total Records: {df.shape[0]}
   • Features: {df.shape[1]}
   • Time Period: {df['Date'].min().strftime('%Y')} - {df['Date'].max().strftime('%Y')}
   
📈 KEY METRICS:
   • Average Unemployment: {df['Unemployment_Rate'].mean():.2f}%
   • COVID Impact: +{covid_peak.mean() - pre_covid.mean():.2f}%
   • Recovery Status: {post_covid.mean():.2f}% (2022)
   
🎯 ANALYSIS COMPLETED:
   ✓ Data cleaning and preprocessing
   ✓ Exploratory data analysis
   ✓ COVID-19 impact assessment
   ✓ Seasonal trend identification
   ✓ Regional and country-wise analysis
   ✓ Policy recommendations generated

✅ Project ready for submission!
""")
print("="*70)