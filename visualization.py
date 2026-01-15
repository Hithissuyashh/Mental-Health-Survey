import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================
# Load data
# ============================
df = pd.read_csv("data/survey.csv")


# ============================
# Charts path (ABSOLUTE)
# ============================
CHARTS_PATH = r"C:\Users\Admin\OneDrive\Desktop\mental-health-survey-visualization\charts"
os.makedirs(CHARTS_PATH, exist_ok=True)


# ============================
# Helper functions
# ============================
sns.set(style="whitegrid")


def save_plot(title, filename, dpi=150):
    """Finalize, save and close current matplotlib figure."""
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(CHARTS_PATH, filename)
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved: {filename}")


def safe_series(col):
    """Return a cleaned series if column exists, else None."""
    if col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            return None
        return s
    return None


print("Starting chart generation...")


# ----------------------------
# 1. Histogram – Age Distribution
# ----------------------------
ages = safe_series('Age')
if ages is not None:
    plt.figure(figsize=(8, 5))
    sns.histplot(ages, bins=20, kde=True, color='#4C72B0')
    plt.xlabel('Age')
    plt.ylabel('Count')
    save_plot('01 - Age Distribution of Respondents', '01_hist_age.png')
else:
    print('Skipping Age histogram (no data)')


# ----------------------------
# 2. Countplot – Gender Distribution (counts)
# ----------------------------
gender = safe_series('Gender')
if gender is not None:
    # normalize gender values into two categories: Male, Female
    g = df['Gender'].fillna('').astype(str).str.strip().str.lower()
    is_female = g.str.contains(r'female|woman|f\b', regex=True)
    # 'male' can appear in 'female', so check female first
    is_male = g.str.contains(r'male|man|m\b|cis male|cis-man', regex=True) & ~is_female
    gender_norm = pd.Series(index=df.index, dtype=object)
    gender_norm[is_male.fillna(False)] = 'Male'
    gender_norm[is_female.fillna(False)] = 'Female'

    # focus only on Male and Female
    df_gender = df.loc[gender_norm.isin(['Male', 'Female'])].copy()
    df_gender['gender_norm'] = gender_norm.loc[df_gender.index]

    counts = df_gender['gender_norm'].value_counts().reindex(['Male', 'Female']).fillna(0).astype(int)
    total = counts.sum()
    if total == 0:
        print('No Male/Female entries to plot for Gender.')
    else:
        pct = (counts / total * 100).round(1)
        # treatment counts (safe)
        if 'treatment' in df_gender.columns:
            treat_yes = df_gender['treatment'].fillna('').astype(str).str.strip().str.lower() == 'yes'
            treat_counts = df_gender.loc[treat_yes].groupby('gender_norm').size().reindex(['Male', 'Female']).fillna(0).astype(int)
        else:
            treat_counts = pd.Series([0, 0], index=['Male', 'Female'])

        # colors mapped to treatment counts
        import matplotlib as mpl
        cmap = plt.cm.OrRd
        norm = mpl.colors.Normalize(vmin=0, vmax=max(treat_counts.max(), 1))
        colors = [mpl.colors.to_hex(cmap(norm(v))) for v in treat_counts.values]

        plt.figure(figsize=(7, 4))
        ax = plt.gca()
        y = list(range(len(counts)))
        ax.barh(y, counts.values, color=colors, edgecolor='k')
        ax.set_yticks(y)
        ax.set_yticklabels(counts.index)
        ax.set_xlabel('Count')
        ax.set_title('02 - Gender Distribution (Male / Female)')

        # annotate count and percentage and treatment count
        for i, (c, p, t) in enumerate(zip(counts.values, pct.values, treat_counts.values)):
            ax.text(c + max(1, total * 0.01), i, f'{c} ({p}%)', va='center')
            ax.text(max(1, total * 0.02), i - 0.25, f'Treatment yes: {t}', va='center', color='black', fontsize=9)

        # color legend for treatment counts
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Number of people who took treatment (Yes)')

        save_plot('02 - Gender Distribution (Countplot)', '02_bar_gender.png')
else:
    print('Skipping Gender chart')


# ----------------------------
# 3. Bar Chart – Top Countries
# ----------------------------
country = safe_series('Country')
if country is not None:
    plt.figure(figsize=(8, 5))
    top = country.value_counts().head(10)
    sns.barplot(x=top.values, y=top.index, palette='muted')
    plt.xlabel('Responses')
    save_plot('03 - Top 10 Countries by Responses', '03_bar_country.png')
else:
    print('Skipping Country chart')


# ----------------------------
# 4. Pie/Donut Chart – Family History
# ----------------------------
fam = safe_series('family_history')
if fam is not None:
    plt.figure(figsize=(6, 6))
    counts = fam.value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.45))
    save_plot('04 - Family History of Mental Illness (Donut)', '04_pie_family_history.png')
else:
    print('Skipping Family History chart')


# ----------------------------
# 5. Bar Chart – Treatment Taken
# ----------------------------
treat = safe_series('treatment')
if treat is not None:
    plt.figure(figsize=(6, 4))
    vc = treat.value_counts(normalize=True) * 100
    sns.barplot(x=vc.index, y=vc.values, palette=['#2ca02c', '#d62728', '#9467bd'])
    plt.ylabel('Percentage (%)')
    save_plot('05 - Mental Health Treatment Taken (%)', '05_bar_treatment.png')
else:
    print('Skipping Treatment chart')


# ----------------------------
# 6. Bar Chart – Work Interference
# ----------------------------
work = safe_series('work_interfere')
order = ['Never', 'Rarely', 'Sometimes', 'Often']
if work is not None:
    plt.figure(figsize=(6, 4))
    vc = work.value_counts().reindex(order).dropna()
    sns.barplot(x=vc.index, y=vc.values, palette='Blues')
    plt.ylabel('Count')
    save_plot('06 - Mental Health Interfering with Work', '06_bar_work_interfere.png')
else:
    print('Skipping Work Interference chart')


# ----------------------------
# 7. Bar Chart – Company Size
# ----------------------------
company = safe_series('no_employees')
if company is not None:
    plt.figure(figsize=(7, 4))
    vc = company.value_counts()
    sns.barplot(x=vc.index, y=vc.values, palette='coolwarm')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    save_plot('07 - Company Size Distribution', '07_bar_company_size.png')
else:
    print('Skipping Company Size chart')


# ----------------------------
# 8. Donut Chart – Remote Work
# ----------------------------
remote = safe_series('remote_work')
if remote is not None:
    plt.figure(figsize=(6, 6))
    counts = remote.value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.45))
    save_plot('08 - Remote Work Availability (Donut)', '08_donut_remote_work.png')
else:
    print('Skipping Remote Work chart')


# ----------------------------
# 9. Bar Chart – Mental Health Benefits
# ----------------------------
benefits = safe_series('benefits')
if benefits is not None:
    plt.figure(figsize=(6, 4))
    vc = benefits.value_counts()
    sns.barplot(x=vc.index, y=vc.values, palette='Set2')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    save_plot('09 - Availability of Mental Health Benefits', '09_bar_benefits.png')
else:
    print('Skipping Benefits chart')


# ----------------------------
# 10. Bar Chart – Wellness Programs
# ----------------------------
wellness = safe_series('wellness_program')
if wellness is not None:
    plt.figure(figsize=(6, 4))
    vc = wellness.value_counts()
    sns.barplot(x=vc.index, y=vc.values, palette='Set3')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    save_plot('10 - Workplace Wellness Programs', '10_bar_wellness.png')
else:
    print('Skipping Wellness chart')


# ----------------------------
# 11. Stripplot – Age distribution as jittered points (replacement for line trend)
# ----------------------------
if ages is not None:
    plt.figure(figsize=(10, 3))
    ages_sorted = ages.sort_values().reset_index(drop=True)
    df_line = ages_sorted.to_frame(name='Age')
    df_line['category'] = 'All'

    # horizontal stripplot to show individual ages with jitter
    sns.stripplot(x='Age', y='category', data=df_line, orient='h', jitter=0.25, size=4, alpha=0.6, color='#1f77b4')

    # overlay a horizontal boxplot to show quartiles/median
    sns.boxplot(x='Age', y='category', data=df_line, width=0.2, showcaps=True,
                boxprops={'facecolor':'none', 'edgecolor':'k'}, showfliers=False)

    # annotate summary statistics
    s = ages_sorted.dropna()
    mean_age = s.mean()
    median_age = s.median()
    plt.xlabel('Age')
    plt.yticks([])
    plt.title('11 - Age Distribution (Stripplot with Box)')
    plt.gca().text(0.98, 0.7, f'Mean: {mean_age:.1f}\nMedian: {median_age:.1f}',
                   transform=plt.gca().transAxes, ha='right', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    save_plot('11 - Age Distribution (Stripplot)', '11_line_age_trend.png')
else:
    print('Skipping Age trend chart')


# ----------------------------
# 12. Violin + Swarm – Age distribution by Work Interference (alternative to scatter)
# ----------------------------
if ages is not None and work is not None:
    plt.figure(figsize=(9, 5))
    work_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
    scatter_df = df.loc[df['work_interfere'].isin(work_map.keys()) & df['Age'].notna(), ['Age', 'work_interfere', 'treatment']].copy()
    order = ['Never', 'Rarely', 'Sometimes', 'Often']
    sns.violinplot(x='work_interfere', y='Age', data=scatter_df, order=order, inner='quartile', palette='Set2')
    # overlay individual points
    sns.swarmplot(x='work_interfere', y='Age', data=scatter_df, order=order, color='k', alpha=0.5, size=3)
    plt.xlabel('Work Interference')
    plt.ylabel('Age')
    save_plot('12 - Age distribution by Work Interference (Violin + Swarm)', '12_scatter_age_work.png')
else:
    print('Skipping Violin/Swarm plot')


# ----------------------------
# 13. Violin + Boxplot – Age distribution (showing skewness and median)
# ----------------------------
if ages is not None:
    plt.figure(figsize=(8, 5))
    # violin to show distribution shape and boxplot for quartiles
    sns.violinplot(y=ages, inner=None, color='#9ecae1')
    sns.boxplot(y=ages, width=0.12, showcaps=True, boxprops={'facecolor':'none'}, showfliers=False)
    s = ages.dropna()
    skewness = s.skew()
    median = s.median()
    plt.ylabel('Age')
    plt.title('13 - Age Distribution (Violin + Boxplot)')
    # annotate skewness and median
    plt.gca().text(0.95, 0.95, f'Skewness: {skewness:.2f}\nMedian: {median:.1f}',
                     transform=plt.gca().transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    save_plot('13 - Age Distribution (Violin + Boxplot)', '13_area_age.png')
else:
    print('Skipping Age distribution (violin) chart')


# ----------------------------
# 14. Box Plot – Age vs Treatment
# ----------------------------
if 'treatment' in df.columns and 'Age' in df.columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='treatment', y='Age', data=df, showfliers=False, palette='Paired')
    plt.xlabel('Treatment')
    plt.ylabel('Age')
    save_plot('14 - Age vs Treatment (Box Plot)', '14_box_age_treatment.png')
else:
    print('Skipping Box plot')


# ----------------------------
# 15. Bubble Chart – Age vs Work Interference (size by Age)
# ----------------------------
if ages is not None and work is not None:
    plt.figure(figsize=(8, 5))
    bubble_df = df.loc[df['Age'].notna() & df['work_interfere'].isin(work_map.keys()), ['Age', 'work_interfere', 'treatment']].copy()
    bubble_df['work_num'] = bubble_df['work_interfere'].map(work_map)
    # safe sizes
    sizes = (bubble_df['Age'] - bubble_df['Age'].min() + 1).clip(lower=5)
    plt.scatter(bubble_df['Age'], bubble_df['work_num'], s=sizes * 3, alpha=0.5, c=sizes, cmap='viridis')
    plt.yticks(list(work_map.values()), list(work_map.keys()))
    plt.xlabel('Age')
    plt.ylabel('Work Interference')
    save_plot('15 - Age vs Work Interference (Bubble Chart)', '15_bubble_age_work.png')
else:
    print('Skipping Bubble chart')


# ----------------------------
# 16. Heatmap – Correlation Matrix (encode relevant categoricals)
# ----------------------------
# Create a working dataframe with numeric columns and safe encodings for categorical
enc = df.select_dtypes(include=[np.number]).copy()
# mappings for common categorical columns
cat_maps = {
    'work_interfere': {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3},
    'treatment': {'No': 0, 'Yes': 1},
    'family_history': {'No': 0, 'Yes': 1},
    'remote_work': {'No': 0, 'Yes': 1},
    'benefits': {'No': 0, 'Yes': 1}
}
for col, mapping in cat_maps.items():
    if col in df.columns:
        try:
            enc[col] = df[col].map(mapping)
        except Exception:
            # fallback: factorize
            enc[col] = pd.factorize(df[col].fillna(''))[0]

if enc.shape[1] >= 1:
    plt.figure(figsize=(10, 8))
    corr = enc.corr()
    # mask upper triangle for clarity
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                cbar_kws={'shrink': .75}, linewidths=0.5)
    save_plot('16 - Correlation Heatmap (encoded)', '16_heatmap_correlation.png')
else:
    print('Skipping Heatmap (no numeric or encodable columns)')

# ----------------------------
# 17. Pairplot – pairwise relationships for a small set of variables (sampled)
# ----------------------------
pp_cols = []
if 'Age' in df.columns:
    pp_cols.append('Age')
for col in ['work_interfere', 'treatment', 'family_history', 'benefits', 'remote_work']:
    if col in df.columns:
        pp_cols.append(col)

pp_cols = pp_cols[:6]
if len(pp_cols) >= 2:
    pp_df = df[pp_cols].copy()
    # encode ordered mapping for work_interfere
    if 'work_interfere' in pp_df.columns:
        pp_df['work_interfere'] = pp_df['work_interfere'].map({'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3})
    # convert simple yes/no to 1/0 where applicable
    for c in pp_df.columns:
        if c != 'Age' and pp_df[c].dtype == object:
            pp_df[c] = pp_df[c].apply(lambda v: 1 if str(v).strip().lower() == 'yes' else (0 if str(v).strip().lower() == 'no' else v))

    pp_df = pp_df.dropna()
    if not pp_df.empty:
        sample_n = min(len(pp_df), 300)
        pp_df_sample = pp_df.sample(n=sample_n, random_state=0)
        hue = None
        if 'treatment' in pp_df_sample.columns:
            # keep treatment as categorical for hue
            pp_df_sample['treatment'] = pp_df_sample['treatment'].apply(lambda v: 'Yes' if str(v) == '1' or str(v).strip().lower() == 'yes' else ('No' if str(v) == '0' or str(v).strip().lower() == 'no' else str(v)))
            hue = 'treatment'
        try:
            g = sns.pairplot(pp_df_sample, hue=hue, diag_kind='kde', corner=False)
            out = os.path.join(CHARTS_PATH, '17_pairplot.png')
            g.fig.savefig(out, dpi=150)
            plt.close()
            print('Saved: 17_pairplot.png')
        except Exception as e:
            print('Pairplot failed:', e)
    else:
        print('Skipping pairplot (no data after dropna)')
else:
    print('Skipping pairplot (not enough columns)')

print('Chart generation complete.')
