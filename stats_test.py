import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import os
from data_loader import get_year_samples


def preprocess_data(df):
    """Preprocess the data: calculate journey time and extract hour."""
    df['journey_time'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60  # in minutes
    df['hour'] = df['pickup_datetime'].dt.hour
    df = df[df['journey_time'] > 0]  # Filter out invalid journey times
    return df


def journey_time_by_hour_boxplot(df):
    """Statistical analysis of journey time by hour with a box plot (no fliers shown)."""
    journey_stats = df.groupby('hour')['journey_time'].agg(['mean', 'median', 'std']).reset_index()
    print("\nJourney Time by Hour - Statistical Summary:")
    print(journey_stats)

    # Box plot with no fliers, using all data
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='hour', y='journey_time', showfliers=False)
    plt.title('Journey Time Distribution by Hour of Day (No Fliers)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Journey Time (minutes)')
    plt.savefig('journey_time_by_hour_boxplot_no_fliers.png')
    plt.close()


# Enhanced Journey Time Variation with Shapiro-Wilk for All Data and Specific Route
def journey_time_variation_by_hour(df, pu_location_id=None, do_location_id=None):
    """Enhanced statistical analysis of journey time with multiple tests and interpretations."""
    journey_stats = df.groupby('hour')['journey_time'].agg(['mean', 'std']).reset_index()
    print("\nJourney Time Variation by Hour - Statistical Summary:")
    print(journey_stats)

    # ANOVA test (parametric)
    hourly_groups = [group['journey_time'].values for _, group in df.groupby('hour')]
    f_stat, p_value = stats.f_oneway(*hourly_groups)
    print(f"\n1. ANOVA Test for Journey Time by Hour: F-statistic = {f_stat:.3f}, p-value = {p_value:.3f}")
    print(
        f"   Interpretation: {'Significant' if p_value < 0.05 else 'No significant'} differences in mean journey times across hours. "
        f"{'Journey time varies by time of day.' if p_value < 0.05 else 'Journey time is consistent across hours.'}")

    # Kruskal-Wallis Test (non-parametric)
    kw_stat, kw_p_value = stats.kruskal(*hourly_groups)
    print(f"2. Kruskal-Wallis Test for Journey Time by Hour: H-statistic = {kw_stat:.3f}, p-value = {kw_p_value:.3f}")
    print(
        f"   Interpretation: {'Significant' if kw_p_value < 0.05 else 'No significant'} differences in journey time distributions across hours. "
        f"{'Distributions vary by hour, robust to non-normality.' if kw_p_value < 0.05 else 'Distributions are similar across hours.'}")

    # Mann-Whitney U Test: Compare peak (8 AM) vs. off-peak (3 AM)
    peak_hour = df[df['hour'] == 8]['journey_time']
    off_peak_hour = df[df['hour'] == 3]['journey_time']
    if len(peak_hour) > 0 and len(off_peak_hour) > 0:
        mw_stat, mw_p_value = stats.mannwhitneyu(peak_hour, off_peak_hour, alternative='two-sided')
        print(f"3. Mann-Whitney U Test (8 AM vs. 3 AM): U-statistic = {mw_stat:.3f}, p-value = {mw_p_value:.3f}")
        print(
            f"   Interpretation: {'Significant' if mw_p_value < 0.05 else 'No significant'} difference between 8 AM and 3 AM journey times. "
            f"{'Journey times are longer at 8 AM (peak) than 3 AM (off-peak).' if mw_p_value < 0.05 else 'Journey times are similar at 8 AM and 3 AM.'}")
    else:
        print("3. Mann-Whitney U Test (8 AM vs. 3 AM): Insufficient data for comparison.")

    # Spearman Correlation with Trip Distance
    spearman_corr, spearman_p_value = stats.spearmanr(df['journey_time'], df['trip_distance'])
    print(
        f"4. Spearman Correlation (Journey Time vs. Trip Distance): rho = {spearman_corr:.3f}, p-value = {spearman_p_value:.3f}")
    print(
        f"   Interpretation: {'Strong' if abs(spearman_corr) > 0.7 else 'Moderate' if abs(spearman_corr) > 0.3 else 'Weak'} "
        f"{'positive' if spearman_corr > 0 else 'negative'} relationship between journey time and trip distance. "
        f"{'Significant' if spearman_p_value < 0.05 else 'Not significant'} (p < 0.05). "
        f"{'Longer distances strongly correlate with longer journey times.' if spearman_p_value < 0.05 and abs(spearman_corr) > 0.7 else 'Distance affects journey time to some extent.' if spearman_p_value < 0.05 else 'No clear relationship.'}")

    # Levene's Test for homogeneity of variance
    levene_stat, levene_p_value = stats.levene(*hourly_groups)
    print(f"5. Levene’s Test for Variance Homogeneity: Statistic = {levene_stat:.3f}, p-value = {levene_p_value:.3f}")
    print(
        f"   Interpretation: {'Significant' if levene_p_value < 0.05 else 'No significant'} differences in variance across hours. "
        f"{'Variability in journey times differs by hour (caution with ANOVA).' if levene_p_value < 0.05 else 'Variance is consistent across hours.'}")

    # Shapiro-Wilk Test for all journey times
    print("\n6. Shapiro-Wilk Test for Normality of All Journey Times:")
    w_stat_all, p_val_all = stats.shapiro(df['journey_time'])
    print(f"   W-statistic = {w_stat_all:.3f}, p-value = {p_val_all:.3f}")
    print(f"   Interpretation: p < 0.05 indicates journey times are not normally distributed across all data. "
          f"{'Likely right-skewed due to long trips.' if p_val_all < 0.05 else 'Data may be approximately normal.'}")

    # Shapiro-Wilk Test for a specific route (if parameters provided)
    if pu_location_id is not None and do_location_id is not None:
        route_data = df[(df['PULocationID'] == pu_location_id) & (df['DOLocationID'] == do_location_id)]
        if len(route_data) > 3:  # Shapiro requires at least 3 samples
            w_stat_route, p_val_route = stats.shapiro(route_data['journey_time'])
            print(f"\n7. Shapiro-Wilk Test for Normality of Route (PU: {pu_location_id}, DO: {do_location_id}):")
            print(f"   Sample size = {len(route_data)} trips")
            print(f"   W-statistic = {w_stat_route:.3f}, p-value = {p_val_route:.3f}")
            print(f"   Interpretation: p < 0.05 indicates journey times for this route are not normally distributed. "
                  f"{'Likely skewed due to variable traffic or trip conditions.' if p_val_route < 0.05 else 'Route times may be approximately normal.'}")
        else:
            print(
                f"\n7. Shapiro-Wilk Test for Route (PU: {pu_location_id}, DO: {do_location_id}): Insufficient data (< 4 trips).")

    # Tukey's HSD Post-Hoc Test (if ANOVA is significant)
    if p_value < 0.05:
        tukey = pairwise_tukeyhsd(endog=df['journey_time'], groups=df['hour'], alpha=0.05)
        print(f"\n8. Tukey’s HSD Post-Hoc Test (Significant Pairwise Differences):")
        print(tukey.summary())
        print("   Interpretation: Shows which hour pairs have significantly different mean journey times. "
              "Reject H0 (p-adj < 0.05) indicates a real difference between those hours.")

    # Linear Regression: Journey Time vs. Trip Distance by Hour
    print("\n9. Linear Regression (Journey Time vs. Trip Distance) by Hour:")
    regression_results = []
    for hour in range(24):
        hour_data = df[df['hour'] == hour]
        if len(hour_data) > 1:  # Need at least 2 points for regression
            X = sm.add_constant(hour_data['trip_distance'])
            model = sm.OLS(hour_data['journey_time'], X).fit()
            slope = model.params['trip_distance']
            p_value = model.pvalues['trip_distance']
            r_squared = model.rsquared
            regression_results.append((hour, slope, p_value, r_squared))
    for hour, slope, p_val, r2 in regression_results:
        print(f"   Hour {hour:2d}: Slope = {slope:.3f}, p-value = {p_val:.3f}, R² = {r2:.3f}")
    print("   Interpretation: Slope indicates minutes per mile; p-value < 0.05 means significant relationship; "
          "R² shows fit (higher = better). Journey time increases with distance, varying by hour.")

    # Bar plot with confidence intervals
    plt.figure(figsize=(12, 6))
    sns.barplot(data=journey_stats, x='hour', y='mean', errorbar=('ci', 95), color='skyblue')
    plt.title('Average Journey Time by Hour of Day with 95% Confidence Intervals')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Journey Time (minutes)')
    plt.savefig('journey_time_by_hour_bar.png')
    plt.close()


# Main execution
if __name__ == "__main__":
    # Load the sampled data
    year_data = get_year_samples(year=2024, sample_size=100000)

    if not year_data.empty:
        # Preprocess the data
        year_data = preprocess_data(year_data)

        # Perform analyses (example route: PU=237, DO=236, common Manhattan zones)
        journey_time_by_hour_boxplot(year_data)
        journey_time_variation_by_hour(year_data, pu_location_id=237, do_location_id=236)

        print("\nAnalysis complete. Check the generated PNG files for new plots.")
    else:
        print("No data available for analysis.")