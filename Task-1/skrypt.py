import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# Ustawienia wizualne
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 120,
})
sns.set_style('whitegrid')

PALETTE = {'Male': '#2563EB', 'Female': '#DB2777'}
PLOTS_DIR = Path('./plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig, filename):
    out_path = PLOTS_DIR / filename
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Zapisano wykres: {out_path}')
    plt.close(fig)


print('Biblioteki załadowane.')

# 3. Wczytanie i przygotowanie danych
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week',
        'native_country', 'income']

url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
       'adult/adult.data')

try:
    df = pd.read_csv(url, header=None, names=cols,
                     na_values=' ?', skipinitialspace=True)
    print(f'Dane wczytane z UCI: {len(df):,} wierszy')
except Exception:
    print('Brak połączenia z UCI — generowanie danych syntetycznych...')
    rng = np.random.default_rng(42)
    n = 32561
    age = rng.integers(17, 90, n)
    sex = rng.choice(['Male', 'Female'], n, p=[0.67, 0.33])
    ednum = rng.integers(1, 17, n)
    hpw = rng.integers(1, 99, n)
    cap_gain = np.where(rng.random(n) < 0.08, rng.integers(1, 99999, n), 0)
    cap_loss = np.where(rng.random(n) < 0.05, rng.integers(1, 4356, n), 0)
    p_inc = np.clip(
        0.18 + 0.025 * (ednum - 9) + 0.003 * (age - 38)
        + np.where(sex == 'Male', 0.08, -0.04), 0.03, 0.95)
    income = np.where(rng.random(n) < p_inc, '>50K', '<=50K')
    wc_p = np.array([0.69, 0.08, 0.03, 0.03, 0.06, 0.04, 0.01, 0.006])
    wc_p /= wc_p.sum()
    wc = rng.choice(['Private', 'Self-emp-not-inc', 'Self-emp-inc',
                     'Federal-gov', 'Local-gov', 'State-gov',
                     'Without-pay', 'Never-worked'], n, p=wc_p)
    occ = rng.choice(['Tech-support', 'Craft-repair', 'Other-service',
                      'Sales', 'Exec-managerial', 'Prof-specialty',
                      'Handlers-cleaners', 'Machine-op-inspct',
                      'Adm-clerical', 'Farming-fishing',
                      'Transport-moving', 'Priv-house-serv',
                      'Protective-serv', 'Armed-Forces'], n)
    edu = rng.choice(['Bachelors', 'Some-college', '11th', 'HS-grad',
                      'Prof-school', 'Assoc-acdm', 'Assoc-voc',
                      '9th', '7th-8th', '12th', 'Masters',
                      '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'], n)
    ms = rng.choice(['Married-civ-spouse', 'Divorced', 'Never-married',
                     'Separated', 'Widowed', 'Married-spouse-absent',
                     'Married-AF-spouse'], n,
                    p=[0.46, 0.13, 0.33, 0.03, 0.04, 0.005, 0.005])
    df = pd.DataFrame({'age': age, 'workclass': wc,
                       'education': edu, 'education_num': ednum,
                       'marital_status': ms, 'occupation': occ,
                       'sex': sex, 'capital_gain': cap_gain,
                       'capital_loss': cap_loss,
                       'hours_per_week': hpw, 'income': income})

# Czyszczenie danych
df.dropna(inplace=True)
df['sex'] = df['sex'].str.strip()
df['income'] = df['income'].str.strip()
df['income_bin'] = (df['income'] == '>50K').astype(int)

print(f'Po czyszczeniu: {len(df):,} wierszy, {df.shape[1]} kolumn')
print(f"  Mężczyźni: {(df['sex'] == 'Male').sum():,}")
print(f"  Kobiety:   {(df['sex'] == 'Female').sum():,}")

# 4. Statystyki opisowe
num_vars = ['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss']

desc = df[num_vars].describe().T
desc['skewness'] = df[num_vars].skew()
desc['kurtosis'] = df[num_vars].kurt()
desc.columns = ['n', 'średnia', 'std', 'min', 'Q1', 'mediana', 'Q3', 'max', 'skośność', 'kurtoza']
desc.index = ['Wiek', 'Lata edukacji', 'Godz./tydzień', 'Zysk kapital.', 'Strata kapital.']

print('=== Statystyki opisowe zmiennych ilościowych ===')
print(desc.round(2))

print('=== Statystyki według płci ===')
grp = df.groupby('sex')[['age', 'education_num', 'hours_per_week']].agg(
    ['mean', 'median', 'std']
).round(2)
grp.columns = ['_'.join(c) for c in grp.columns]
print(grp)

# Odsetek dochodu >50K według płci
p_income = df.groupby('sex')['income_bin'].agg(['mean', 'count'])
p_income.columns = ['odsetek >50K', 'n']
p_income['odsetek >50K'] = (p_income['odsetek >50K'] * 100).round(1)

print('=== Odsetek dochodów >50K USD/rok według płci ===')
print(p_income)

# Test chi-kwadrat
ct = pd.crosstab(df['sex'], df['income'])
chi2, p_val, dof, _ = stats.chi2_contingency(ct)
print('\nTest chi-kwadrat (płeć x dochód):')
print(f'  chi2 = {chi2:.2f},  df = {dof},  p = {p_val:.2e}')
print(f'  => Różnica jest {"ISTOTNA statystycznie" if p_val < 0.05 else "nieistotna"} (alpha = 0.05)')

# 5.1 Rozkład wieku i poziomu wykształcenia według płci
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
for s, color in PALETTE.items():
    subset = df[df['sex'] == s]['age']
    ax.hist(subset, bins=30, alpha=0.6, color=color, label=s, density=True)
ax.set_xlabel('Wiek')
ax.set_ylabel('Gęstość')
ax.set_title('Rozkład wieku według płci')
ax.legend()

ax = axes[1]
sns.violinplot(data=df, x='sex', y='education_num',
               palette=PALETTE, order=['Male', 'Female'],
               inner='quartile', ax=ax, linewidth=0.8)
ax.set_xlabel('Płeć')
ax.set_ylabel('Poziom wykształcenia (lata)')
ax.set_title('Rozkład poziomu wykształcenia według płci')

plt.tight_layout()
save_plot(fig, 'plot_5_1_age_education_by_sex.png')

# 5.2 Tygodniowy czas pracy i odsetek dochodów >50K według płci
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
sns.boxplot(data=df, x='sex', y='hours_per_week', hue='income',
            palette={'<=50K': '#93C5FD', '>50K': '#1E3A5F'},
            order=['Male', 'Female'], ax=ax, width=0.5, linewidth=0.8,
            flierprops=dict(marker='o', markersize=2, alpha=0.3))
ax.set_xlabel('Płeć')
ax.set_ylabel('Godziny pracy / tydzień')
ax.set_title('Tygodniowy czas pracy wg płci i dochodu')
ax.legend(title='Dochód')

ax = axes[1]
colors_bar = [PALETTE[s] for s in p_income.index]
bars = ax.bar(p_income.index, p_income['odsetek >50K'],
              color=colors_bar, edgecolor='white', width=0.5)
for bar, val in zip(bars, p_income['odsetek >50K']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_ylim(0, p_income['odsetek >50K'].max() * 1.25)
ax.set_ylabel('Odsetek osób z dochodem >50K [%]')
ax.set_title('Odsetek osób zarabiających >50K USD/rok\nwedług płci')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
save_plot(fig, 'plot_5_2_hours_and_income_by_sex.png')

m_med = df[df['sex'] == 'Male']['hours_per_week'].median()
f_med = df[df['sex'] == 'Female']['hours_per_week'].median()
print(f'Mediana godzin pracy: mężczyźni = {m_med:.0f} h, kobiety = {f_med:.0f} h')

# 5.3 Macierz korelacji
fig, ax = plt.subplots(figsize=(7, 5.5))

corr_vars = ['age', 'education_num', 'hours_per_week',
             'capital_gain', 'capital_loss', 'income_bin']
corr_labels = ['Wiek', 'Lata edukacji', 'Godz./tyg.',
               'Zysk kapital.', 'Strata kapital.', 'Dochód >50K']

corr = df[corr_vars].corr()
corr.index = corr_labels
corr.columns = corr_labels

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 10})
ax.set_title('Macierz korelacji zmiennych numerycznych')
plt.tight_layout()
save_plot(fig, 'plot_5_3_correlation_matrix.png')

# 5.4 Zawody według odsetka dochodów >50K
if 'occupation' in df.columns:
    occ_stat = (
        df.groupby('occupation')['income_bin']
        .agg(['mean', 'count'])
        .query('count >= 200')
        .sort_values('mean', ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(occ_stat.index, occ_stat['mean'] * 100,
                   color='#1E3A5F', edgecolor='white')
    for bar, val in zip(bars, occ_stat['mean'] * 100):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.set_xlabel('Odsetek osób z dochodem >50K [%]')
    ax.set_title('Odsetek dochodów >50K według zawodu')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_plot(fig, 'plot_5_4_income_share_by_occupation.png')

# 5.5 Analiza porównawcza — dochód według płci i wykształcenia
edu_bins = [0, 8, 10, 13, 16]
edu_labels = ['Podstawowe\n(1–8)', 'Średnie\n(9–10)', 'Wyższe I st.\n(11–13)', 'Wyższe II/III\n(14–16)']
df['edu_group'] = pd.cut(df['education_num'], bins=edu_bins, labels=edu_labels)

pivot = df.groupby(['edu_group', 'sex'])['income_bin'].mean().unstack() * 100

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(pivot))
w = 0.35
ax.bar(x - w / 2, pivot.get('Male', [0] * len(pivot)),
       width=w, color=PALETTE['Male'], label='Mężczyźni', alpha=0.85)
ax.bar(x + w / 2, pivot.get('Female', [0] * len(pivot)),
       width=w, color=PALETTE['Female'], label='Kobiety', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(pivot.index)
ax.set_ylabel('Odsetek dochodów >50K [%]')
ax.set_title('Odsetek dochodów >50K według płci i poziomu wykształcenia')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_plot(fig, 'plot_5_5_income_by_sex_and_education.png')

print('Odsetek dochodów >50K wg płci i wykształcenia [%]:')
print(pivot.round(1))

print('\nWykresy zapisane w ./plots/')
