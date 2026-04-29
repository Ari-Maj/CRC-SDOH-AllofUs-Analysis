import pandas as pd
import numpy as np
import os
from IPython.display import display
import statsmodels.api as sm


cases_df = pd.read_csv("crc_cases_cohort.csv")

print("Cases loaded:", cases_df.shape)
display(cases_df.head())
print(cases_df.columns.tolist())

cases_df['CRC_status'] = 1
cases_df['match_age'] = cases_df['age_at_diagnosis'].round()

print(cases_df['CRC_status'].value_counts(dropna=False))
display(cases_df[['person_id', 'match_age', 'sex_at_birth']].head())

control_sql = f"""
SELECT
    p.person_id,
    cb.age_at_consent,
    sex.concept_name AS sex_at_birth,
    race.concept_name AS race
FROM `{os.environ["WORKSPACE_CDR"]}.cb_search_person` cb
JOIN `{os.environ["WORKSPACE_CDR"]}.person` p
    ON cb.person_id = p.person_id
LEFT JOIN `{os.environ["WORKSPACE_CDR"]}.concept` sex
    ON p.sex_at_birth_concept_id = sex.concept_id
LEFT JOIN `{os.environ["WORKSPACE_CDR"]}.concept` race
    ON p.race_concept_id = race.concept_id
WHERE cb.age_at_consent >= 18
AND cb.has_ehr_data = 1
AND cb.has_array_data = 1
"""

controls_pool = pd.read_gbq(
    control_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

print("Control pool size:", controls_pool.shape)
display(controls_pool.head())

case_ids = set(cases_df['person_id'])
controls_pool = controls_pool[~controls_pool['person_id'].isin(case_ids)].copy()

print("Controls after removing cases:", controls_pool.shape)

overlap = set(cases_df['person_id']).intersection(set(controls_pool['person_id']))
print("Overlap:", len(overlap))

other_cancer_sql = f"""
SELECT DISTINCT
    co.person_id
FROM `{os.environ["WORKSPACE_CDR"]}.condition_occurrence` co
JOIN `{os.environ["WORKSPACE_CDR"]}.concept` c
    ON co.condition_concept_id = c.concept_id
WHERE
    (
        LOWER(c.concept_name) LIKE '%malignant%'
        OR LOWER(c.concept_name) LIKE '%cancer%'
        OR LOWER(c.concept_name) LIKE '%carcinoma%'
        OR LOWER(c.concept_name) LIKE '%adenocarcinoma%'
        OR LOWER(c.concept_name) LIKE '%neoplasm%'
        OR LOWER(c.concept_name) LIKE '%tumor%'
        OR LOWER(c.concept_name) LIKE '%tumour%'
    )
    AND LOWER(c.concept_name) NOT LIKE '%colon%'
    AND LOWER(c.concept_name) NOT LIKE '%rectum%'
    AND LOWER(c.concept_name) NOT LIKE '%rectal%'
    AND LOWER(c.concept_name) NOT LIKE '%colorectal%'
"""
other_cancer_df = pd.read_gbq(
    other_cancer_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

print("People with non-CRC cancer diagnoses:", other_cancer_df.shape)
display(other_cancer_df.head())

other_cancer_ids = set(other_cancer_df['person_id'])
controls_pool = controls_pool[~controls_pool['person_id'].isin(other_cancer_ids)].copy()

print("Controls after excluding other cancers:", controls_pool.shape)

print("Remaining unique controls:", controls_pool['person_id'].nunique())
display(controls_pool.head())

controls_pool['CRC_status'] = 0
controls_pool['match_age'] = controls_pool['age_at_consent'].round()
controls_pool['used'] = False

print(controls_pool['CRC_status'].value_counts(dropna=False))
display(controls_pool[['person_id', 'match_age', 'sex_at_birth']].head())

matched_controls = []

for _, case in cases_df.iterrows():
    eligible_controls = controls_pool[
        (controls_pool['used'] == False) &
        (controls_pool['sex_at_birth'] == case['sex_at_birth']) &
        (abs(controls_pool['match_age'] - case['match_age']) <= 2)
    ]

    if len(eligible_controls) >= 4:
        sampled = eligible_controls.sample(n=4, random_state=42, replace=False)
        controls_pool.loc[sampled.index, 'used'] = True
        matched_controls.append(sampled)

matched_controls_df = pd.concat(matched_controls, ignore_index=True)

print("Matched controls shape:", matched_controls_df.shape)
print("Unique matched controls:", matched_controls_df['person_id'].nunique())
print("CRC labels in matched controls:")
print(matched_controls_df['CRC_status'].value_counts(dropna=False))

case_control_df = pd.concat([cases_df, matched_controls_df], ignore_index=True)

print("Final dataset shape:", case_control_df.shape)
print(case_control_df['CRC_status'].value_counts(dropna=False))
print("Unique participants:", case_control_df['person_id'].nunique())

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from IPython.display import display

case_control_df = case_control_df.drop(
    columns=["median_income", "poverty", "no_health_insurance"],
    errors="ignore"
)

zip_sql = f"""
SELECT
    observation.person_id,
    zip_code.median_income,
    zip_code.fraction_no_health_ins AS no_health_insurance,
    zip_code.fraction_poverty AS poverty,
    zip_code.deprivation_index AS deprivation_index_zip
FROM `{os.environ["WORKSPACE_CDR"]}.zip3_ses_map` zip_code
JOIN `{os.environ["WORKSPACE_CDR"]}.observation` observation
  ON CAST(SUBSTR(observation.value_as_string, 0, STRPOS(observation.value_as_string, '*') - 1) AS INT64) = zip_code.zip3
WHERE observation_source_concept_id = 1585250
"""

sdoh_df = pd.read_gbq(
    zip_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

case_control_df = case_control_df.merge(sdoh_df, on="person_id", how="left")

if "deprivation_index_zip" in case_control_df.columns:
    case_control_df["deprivation_index"] = case_control_df["deprivation_index"].fillna(
        case_control_df["deprivation_index_zip"]
    )
    case_control_df = case_control_df.drop(columns=["deprivation_index_zip"])

display(case_control_df[[
    "person_id", "median_income", "poverty", "no_health_insurance", "deprivation_index"
]].head())

case_control_df = case_control_df.merge(
    sdoh_df,
    on='person_id',
    how='left'
    
)

print(case_control_df.shape)

def pick_col(base):
    for c in [base, f"{base}_y", f"{base}_x"]:
        if c in case_control_df.columns:
            return c
    return None

median_col = pick_col('median_income')
poverty_col = pick_col('poverty')
ins_col = pick_col('no_health_insurance')
dep_col = pick_col('deprivation_index')

print(median_col, poverty_col, ins_col, dep_col)

case_control_df = case_control_df.rename(columns={
    median_col: 'median_income',
    poverty_col: 'poverty',
    ins_col: 'no_health_insurance',
    dep_col: 'deprivation_index'
})

drop_cols = [c for c in [
    'median_income_x','median_income_y',
    'poverty_x','poverty_y',
    'no_health_insurance_x','no_health_insurance_y',
    'deprivation_index_x','deprivation_index_y'
] if c in case_control_df.columns and c not in ['median_income','poverty','no_health_insurance','deprivation_index']]

case_control_df = case_control_df.drop(columns=drop_cols)

display(case_control_df[['median_income','poverty','no_health_insurance','deprivation_index']].head())

case_control_df.to_csv("case_control_dataset.csv", index=False)
import pandas as pd
# test = pd.read_csv("case_control_dataset.csv")
case_control_df
# print("case_control_dataset.csv saved successfully")

case_control_df["sex_at_birth"] = case_control_df["sex_at_birth"].replace({
    "I prefer not to answer": "Other",
    "Intersex": "Other",
    "PMI: Skip": "Other",
    "Sex At Birth: Sex At Birth None Of These": "Other"
})

rare_races = [
    "I prefer not to answer",
    "Middle Eastern or North African",
    "More than one population",
    "Native Hawaiian or Other Pacific Islander",
    "None Indicated",
    "None of these",
    "PMI: Skip"
]

case_control_df["race_clean"] = case_control_df["race"].replace(
    {r: "Other/Unknown" for r in rare_races}
)

print(case_control_df["race_clean"].value_counts(dropna=False))

print(case_control_df['CRC_status'].value_counts(dropna=False))
print(case_control_df[['deprivation_index', 'sex_at_birth', 'race_clean', 'CRC_status']].isna().sum())

model_df = case_control_df.dropna(subset=[
    "deprivation_index",
    "sex_at_birth",
    "race_clean",
    "CRC_status"
]).copy()

model_df["CRC_status"] = pd.to_numeric(model_df["CRC_status"], errors="coerce")

X = pd.get_dummies(
    model_df[["deprivation_index", "sex_at_birth", "race_clean"]],
    drop_first=True
)

X = X.apply(pd.to_numeric, errors="coerce")

for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

X = sm.add_constant(X)
y = pd.to_numeric(model_df["CRC_status"], errors="coerce")

valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

model = sm.Logit(y, X).fit()
print(model.summary())

or_table = pd.DataFrame({
    "Odds Ratio": np.exp(model.params),
    "CI Lower": np.exp(model.conf_int()[0]),
    "CI Upper": np.exp(model.conf_int()[1]),
    "p_value": model.pvalues
})
display(or_table)



# MONICA TEST

import pandas as pd
import plotly.express as px

# Copy and keep only needed columns
plot_df = case_control_df[["race_clean", "CRC_status"]].dropna().copy()

# Make CRC labels easier to read
plot_df["CRC_status_label"] = plot_df["CRC_status"].map({
    0: "Control (No CRC)",
    1: "CRC Case"
})

# Count participants by race and CRC status
race_counts = (
    plot_df.groupby(["race_clean", "CRC_status_label"])
    .size()
    .reset_index(name="count")
)

# Plot
fig = px.bar(
    race_counts,
    x="race_clean",
    y="count",
    color="CRC_status_label",
    barmode="group",
    text="count",
    title="Colorectal Cancer Cases and Controls by Race",
    labels={
        "race_clean": "Race",
        "count": "Number of Participants",
        "CRC_status_label": "Group"
    }
)

fig.update_layout(
    xaxis_title="Race",
    yaxis_title="Number of Participants",
    legend_title="CRC Status"
)

fig.show()





case_control_df['deprivation_quartile'] = pd.qcut(
    case_control_df['deprivation_index'],
    4,
    labels=['Q1_lowest','Q2','Q3','Q4_highest']
)

case_control_df['deprivation_quartile'].value_counts()

case_control_df["deprivation_quartile"] = pd.qcut(
    case_control_df["deprivation_index"],
    4,
    labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]
)

model_df = case_control_df.dropna(subset=[
    "deprivation_quartile",
    "sex_at_birth",
    "race_clean",
    "CRC_status"
]).copy()

model_df["CRC_status"] = pd.to_numeric(model_df["CRC_status"], errors="coerce")

X = pd.get_dummies(
    model_df[["deprivation_quartile", "sex_at_birth", "race_clean"]],
    drop_first=True
)

X = X.apply(pd.to_numeric, errors="coerce")

for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

X = sm.add_constant(X)
y = pd.to_numeric(model_df["CRC_status"], errors="coerce")

valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

model2 = sm.Logit(y, X).fit()
print(model2.summary())

or_table_quartiles = pd.DataFrame({
    "Odds Ratio": np.exp(model2.params),
    "CI Lower": np.exp(model2.conf_int()[0]),
    "CI Upper": np.exp(model2.conf_int()[1]),
    "p_value": model2.pvalues
})
display(or_table_quartiles)

case_control_df['deprivation_quartile'] = pd.qcut(
    case_control_df['deprivation_index'],
    4,
    labels=['Q1_lowest','Q2','Q3','Q4_highest']
)

case_control_df['deprivation_quartile'].value_counts()

import statsmodels.api as sm
import pandas as pd
import numpy as np

model_df = case_control_df.dropna(subset=[
    'deprivation_quartile',
    'sex_at_birth',
    'race_clean',
    'CRC_status'
]).copy()

# Make outcome numeric
model_df['CRC_status'] = pd.to_numeric(model_df['CRC_status'], errors='coerce')

# Build dummy variables
X = pd.get_dummies(
    model_df[['deprivation_quartile', 'sex_at_birth', 'race_clean']],
    drop_first=True
)

# Force all predictors to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Convert bool columns to int if needed
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

# Add intercept
X = sm.add_constant(X)

# Outcome
y = pd.to_numeric(model_df['CRC_status'], errors='coerce')

# Keep complete rows only
valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print(X.dtypes)
print(y.dtype)
print(X.shape, y.shape)
print(y.value_counts())

model2 = sm.Logit(y, X).fit()
print(model2.summary())

params = model2.params
conf = model2.conf_int()

or_table_quartiles = pd.DataFrame({
    "Odds Ratio": np.exp(params),
    "CI Lower": np.exp(conf[0]),
    "CI Upper": np.exp(conf[1]),
    "p-value": model2.pvalues
})

or_table_quartiles

pd.crosstab(case_control_df['sex_at_birth'], case_control_df['CRC_status'])

pd.crosstab(case_control_df['race_clean'], case_control_df['CRC_status'])

case_control_df.groupby('CRC_status')[
    ['median_income','poverty','no_health_insurance','deprivation_index']
].median()

case_control_df.to_csv("final_case_control_dataset.csv", index=False)

or_table.to_csv("logistic_results_continuous.csv")
or_table_quartiles.to_csv("logistic_results_quartiles.csv")

case_control_df.columns.tolist()

[c for c in case_control_df.columns if
 "smok" in c.lower() or
 "bmi" in c.lower() or
 "insur" in c.lower() or
 "educ" in c.lower() or
 "ethnic" in c.lower()]

bmi_sql = f"""
SELECT
    person_id,
    value_as_number AS BMI
FROM `{os.environ["WORKSPACE_CDR"]}.measurement`
WHERE measurement_concept_id = 3038553
"""

measurement_df = pd.read_gbq(
    bmi_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

measurement_df = measurement_df.dropna(subset=["BMI"])
measurement_df = measurement_df.groupby("person_id")["BMI"].mean().reset_index()

case_control_df = case_control_df.merge(measurement_df, on="person_id", how="left")
print(case_control_df[["BMI"]].notna().sum())

dataset_61896282_measurement_sql = """
SELECT
    person_id,
    value_as_number AS BMI
FROM
    `{}.measurement`
WHERE
    measurement_concept_id = 3038553
""".format(os.environ["WORKSPACE_CDR"])

measurement_df = pandas.read_gbq(
    dataset_61896282_measurement_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

measurement_df.head()

measurement_df = measurement_df.dropna(subset=['BMI'])

measurement_df = measurement_df.groupby('person_id')['BMI'].mean().reset_index()

measurement_df.head()

dataset_61896282_smoking_sql = """
SELECT
    person_id,
    question,
    answer AS smoking_status
FROM
    `{}.ds_survey`
WHERE
    LOWER(question) LIKE '%smok%'
""".format(os.environ["WORKSPACE_CDR"])

smoking_df = pandas.read_gbq(
    dataset_61896282_smoking_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

smoking_df.head()

smoking_df['question'].value_counts().head(20)

smoking_df = smoking_df[
    smoking_df['question'] == 'Smoking: Smoke Frequency'
]

smoking_df = smoking_df[['person_id','smoking_status']].drop_duplicates()

smoking_df.head()

smoking_df['smoking_status'].value_counts()

smoking_sql = f"""
SELECT
    person_id,
    question,
    answer AS smoking_status
FROM `{os.environ["WORKSPACE_CDR"]}.ds_survey`
WHERE LOWER(question) LIKE '%smok%'
"""

smoking_df = pd.read_gbq(
    smoking_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

smoking_df = smoking_df[
    smoking_df["question"] == "Smoking: Smoke Frequency"
][["person_id", "smoking_status"]].drop_duplicates()

# clean smoking variable
smoking_df["smoking_clean"] = smoking_df["smoking_status"].replace({
    "Smoke Frequency: Every Day": "Current",
    "Smoke Frequency: Some Days": "Current",
    "Smoke Frequency: Not At All": "Not current",
    "PMI: Skip": "Unknown",
    "PMI: Prefer Not To Answer": "Unknown",
    "PMI: Dont Know": "Unknown"
})

# if some people somehow have multiple rows, keep one
smoking_df = smoking_df.groupby("person_id", as_index=False)["smoking_clean"].first()

# merge into case_control_df
case_control_df = case_control_df.merge(
    smoking_df[["person_id", "smoking_clean"]],
    on="person_id",
    how="left"
)

print(case_control_df["smoking_clean"].value_counts(dropna=False))

# Fix insurance column
if "no_health_insurance" not in case_control_df.columns:
    for col in case_control_df.columns:
        if "insur" in col.lower():
            case_control_df = case_control_df.rename(columns={col: "no_health_insurance"})
            break

# Handle missing smoking (DO NOT drop)
case_control_df["smoking_clean"] = case_control_df["smoking_clean"].fillna("Unknown")

# Build model dataset
model_df = case_control_df.dropna(subset=[
    "deprivation_quartile",
    "sex_at_birth",
    "race_clean",
    "BMI",
    "no_health_insurance",
    "CRC_status"
]).copy()

model_df["CRC_status"] = pd.to_numeric(model_df["CRC_status"], errors="coerce")

# Predictors
X = pd.get_dummies(
    model_df[[
        "deprivation_quartile",
        "sex_at_birth",
        "race_clean",
        "BMI",
        "smoking_clean",
        "no_health_insurance"
    ]],
    drop_first=True
)

X = X.apply(pd.to_numeric, errors="coerce")

for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

X = sm.add_constant(X)

y = pd.to_numeric(model_df["CRC_status"], errors="coerce")

valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print("Final model size:", X.shape, y.shape)

model_final = sm.Logit(y, X).fit()
print(model_final.summary())

# Odds ratios
or_table_final = pd.DataFrame({
    "Odds Ratio": np.exp(model_final.params),
    "CI Lower": np.exp(model_final.conf_int()[0]),
    "CI Upper": np.exp(model_final.conf_int()[1]),
    "p_value": model_final.pvalues
})

display(or_table_final)

model_df = case_control_df.dropna(subset=[
    "deprivation_quartile",
    "sex_at_birth",
    "race_clean",
    "BMI",
    "smoking_clean",
    "no_health_insurance",
    "CRC_status"
]).copy()

model_df["CRC_status"] = pd.to_numeric(model_df["CRC_status"], errors="coerce")

X = pd.get_dummies(
    model_df[[
        "deprivation_quartile",
        "sex_at_birth",
        "race_clean",
        "BMI",
        "smoking_clean",
        "no_health_insurance"
    ]],
    drop_first=True
)

X = X.apply(pd.to_numeric, errors="coerce")

for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

X = sm.add_constant(X)
y = pd.to_numeric(model_df["CRC_status"], errors="coerce")

valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print(X.shape, y.shape)
print(y.value_counts())

model_final = sm.Logit(y, X).fit()
print(model_final.summary())

or_table_final = pd.DataFrame({
    "Odds Ratio": np.exp(model_final.params),
    "CI Lower": np.exp(model_final.conf_int()[0]),
    "CI Upper": np.exp(model_final.conf_int()[1]),
    "p_value": model_final.pvalues
})
display(or_table_final)

import pandas as pd
import numpy as np
import os

survey_search_sql = f"""
SELECT
    question,
    COUNT(*) AS n_rows
FROM `{os.environ["WORKSPACE_CDR"]}.ds_survey`
WHERE
    LOWER(question) LIKE '%education%'
    OR LOWER(question) LIKE '%school%'
    OR LOWER(question) LIKE '%doctor%'
    OR LOWER(question) LIKE '%care%'
    OR LOWER(question) LIKE '%afford%'
    OR LOWER(question) LIKE '%alcohol%'
    OR LOWER(question) LIKE '%drink%'
GROUP BY question
ORDER BY n_rows DESC, question
"""

survey_questions_df = pd.read_gbq(
    survey_search_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

print("Possible survey questions:")
display(survey_questions_df)

import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

# =========================================================
# STEP 1: Pull survey variables
# =========================================================

survey_sql = f"""
SELECT
    person_id,
    question,
    answer
FROM `{os.environ["WORKSPACE_CDR"]}.ds_survey`
WHERE
    question IN (
        'Education Level: Highest Grade',
        'Alcohol: Drink Frequency Past Year'
    )
    OR LOWER(question) LIKE '%doctor%'
    OR LOWER(question) LIKE '%care%'
    OR LOWER(question) LIKE '%afford%'
"""

survey_df = pd.read_gbq(
    survey_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

# =========================================================
# STEP 2: Separate variables
# =========================================================

education_df = survey_df[
    survey_df["question"] == "Education Level: Highest Grade"
][["person_id", "answer"]].rename(columns={"answer": "education_raw"})

alcohol_df = survey_df[
    survey_df["question"] == "Alcohol: Drink Frequency Past Year"
][["person_id", "answer"]].rename(columns={"answer": "alcohol_raw"})

# Healthcare access (broad capture)
access_df = survey_df[
    survey_df["question"].str.contains("doctor|care|afford", case=False, na=False)
][["person_id", "answer"]].rename(columns={"answer": "access_raw"})

# Keep one per person
education_df = education_df.groupby("person_id", as_index=False).first()
alcohol_df = alcohol_df.groupby("person_id", as_index=False).first()
access_df = access_df.groupby("person_id", as_index=False).first()

# =========================================================
# STEP 3: Merge into main dataset
# =========================================================

case_control_df = case_control_df.merge(education_df, on="person_id", how="left")
case_control_df = case_control_df.merge(alcohol_df, on="person_id", how="left")
case_control_df = case_control_df.merge(access_df, on="person_id", how="left")

# =========================================================
# STEP 4: CLEAN VARIABLES
# =========================================================

# -------- EDUCATION --------
def clean_education(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x)

    if any(k in x for k in ["Never", "Grades 1", "Grades 2", "Grades 3", "Grades 4", "Grades 5", "Grades 6", "Grades 7", "Grades 8", "Grades 9", "Grades 10", "Grades 11"]):
        return "Less than HS"
    elif "High school" in x or "GED" in x:
        return "High school"
    elif "college" in x.lower() or "Associate" in x:
        return "Some college"
    elif "Bachelor" in x or "Master" in x or "Doctoral" in x:
        return "College+"
    elif "PMI" in x:
        return "Unknown"
    else:
        return "Other"

case_control_df["education_clean"] = case_control_df["education_raw"].apply(clean_education)

# -------- ALCOHOL --------
def clean_alcohol(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x)

    if "Never" in x:
        return "Never"
    elif "Monthly" in x or "2 To 4 Times A Month" in x:
        return "Occasional"
    elif "2 To 3 Times A Week" in x or "4 Or More Times A Week" in x:
        return "Frequent"
    elif "PMI" in x:
        return "Unknown"
    else:
        return "Other"

case_control_df["alcohol_clean"] = case_control_df["alcohol_raw"].apply(clean_alcohol)

# -------- HEALTHCARE ACCESS --------
def clean_access(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x)

    if "afford" in x.lower() or "could not" in x.lower():
        return "Cost barrier"
    elif "yes" in x.lower() or "able" in x.lower():
        return "No barrier"
    elif "PMI" in x:
        return "Unknown"
    else:
        return "Other"

case_control_df["access_clean"] = case_control_df["access_raw"].apply(clean_access)

# =========================================================
# STEP 5: Check distributions
# =========================================================

print("\nEducation:")
print(case_control_df["education_clean"].value_counts())

print("\nAlcohol:")
print(case_control_df["alcohol_clean"].value_counts())

print("\nHealthcare access:")
print(case_control_df["access_clean"].value_counts())

# =========================================================
# STEP 6: Prepare model
# =========================================================

# fix insurance column name if needed
if "no_health_insurance" not in case_control_df.columns:
    for col in case_control_df.columns:
        if "insur" in col.lower():
            case_control_df = case_control_df.rename(columns={col: "no_health_insurance"})
            break

case_control_df["smoking_clean"] = case_control_df["smoking_clean"].fillna("Unknown")

model_df = case_control_df.dropna(subset=[
    "CRC_status",
    "deprivation_quartile",
    "sex_at_birth",
    "race_clean",
    "BMI",
    "no_health_insurance"
]).copy()

# =========================================================
# STEP 7: Build model
# =========================================================

X = pd.get_dummies(
    model_df[[
        "deprivation_quartile",
        "sex_at_birth",
        "race_clean",
        "BMI",
        "no_health_insurance",
        "smoking_clean",
        "education_clean",
        "access_clean",
        "alcohol_clean"
    ]],
    drop_first=True
)

X = X.apply(pd.to_numeric, errors="coerce")

for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

X = sm.add_constant(X)

y = pd.to_numeric(model_df["CRC_status"], errors="coerce")

valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print("Final model size:", X.shape)

# =========================================================
# STEP 8: Run model
# =========================================================

model_sdoh = sm.Logit(y, X).fit()
print(model_sdoh.summary())

or_table_sdoh = pd.DataFrame({
    "Odds Ratio": np.exp(model_sdoh.params),
    "CI Lower": np.exp(model_sdoh.conf_int()[0]),
    "CI Upper": np.exp(model_sdoh.conf_int()[1]),
    "p_value": model_sdoh.pvalues
})

display(or_table_sdoh)

from scipy.stats import fisher_exact
import pandas as pd

results = []

for cat in case_control_df["education_clean"].dropna().unique():
    if cat != "High school":  # reference group
        res = run_fisher(case_control_df, "education_clean", cat)
        if res:
            results.append(res)
for cat in case_control_df["access_clean"].dropna().unique():
    if cat != "No barrier":
        res = run_fisher(case_control_df, "access_clean", cat)
        if res:
            results.append(res)
            
for cat in case_control_df["alcohol_clean"].dropna().unique():
    if cat != "Never":
        res = run_fisher(case_control_df, "alcohol_clean", cat)
        if res:
            results.append(res)

fisher_results = pd.DataFrame(results)
display(fisher_results.sort_values("p_value"))

cases_only_df = case_control_df[case_control_df["CRC_status"] == 1].copy()

cases_only_df["early_onset"] = np.where(
    cases_only_df["age_at_diagnosis"] <= 50, 1, 0
)

print(cases_only_df["early_onset"].value_counts())

print("Total cases:", len(cases_only_df))
print("Early onset:", cases_only_df["early_onset"].sum())
print("Late onset:", (cases_only_df["early_onset"] == 0).sum())

model_df = cases_only_df.dropna(subset=[
    "early_onset",
    "deprivation_quartile",
    "sex_at_birth",
    "race_clean",
    "BMI",
    "no_health_insurance",
    "smoking_clean",
    "education_clean",
    "access_clean",
    "alcohol_clean"
]).copy()

X = pd.get_dummies(
    model_df[[
        "deprivation_quartile",
        "sex_at_birth",
        "race_clean",
        "BMI",
        "no_health_insurance",
        "smoking_clean",
        "education_clean",
        "access_clean",
        "alcohol_clean"
    ]],
    drop_first=True
)

X = X.apply(pd.to_numeric, errors="coerce")

for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

X = sm.add_constant(X)

y = model_df["early_onset"]

valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print("Model size:", X.shape)
print(y.value_counts())

model_early = sm.Logit(y, X).fit()
print(model_early.summary())

or_table_early = pd.DataFrame({
    "Odds Ratio": np.exp(model_early.params),
    "CI Lower": np.exp(model_early.conf_int()[0]),
    "CI Upper": np.exp(model_early.conf_int()[1]),
    "p_value": model_early.pvalues
})

display(or_table_early)

from scipy.stats import fisher_exact

def fisher_early(df, var, category):
    table = pd.crosstab(df[var] == category, df["early_onset"])
    
    if table.shape != (2,2):
        return None
    
    or_val, p_val = fisher_exact(table)
    
    return {
        "variable": var,
        "category": category,
        "odds_ratio": or_val,
        "p_value": p_val
    }
results = []

for cat in cases_only_df["education_clean"].unique():
    if cat != "High school":
        res = fisher_early(cases_only_df, "education_clean", cat)
        if res:
            results.append(res)

pd.DataFrame(results)

import pandas as pd
from scipy.stats import fisher_exact

# =========================================================
# Helper function:
# Creates a 2x2 contingency table for one category vs all others
# and runs Fisher's exact test
# =========================================================
def contingency_and_fisher(df, variable, category, outcome="CRC_status"):
    temp = df[[variable, outcome]].dropna().copy()
    temp["category_yes"] = temp[variable] == category

    table = pd.crosstab(temp["category_yes"], temp[outcome])

    # force 2x2 shape in case one level is missing
    table = table.reindex(index=[False, True], columns=[0, 1], fill_value=0)

    odds_ratio, p_value = fisher_exact(table)

    print(f"\n==============================")
    print(f"Variable: {variable}")
    print(f"Category tested: {category}")
    print("2x2 contingency table:")
    print(table)
    print(f"Odds Ratio: {odds_ratio:.6f}")
    print(f"P-value: {p_value:.6f}")

    return {
        "variable": variable,
        "category": category,
        "no_crc_other": table.loc[False, 0],
        "crc_other": table.loc[False, 1],
        "no_crc_category": table.loc[True, 0],
        "crc_category": table.loc[True, 1],
        "odds_ratio": odds_ratio,
        "p_value": p_value
    }

# =========================================================
# Optional: full contingency tables for each categorical variable
# =========================================================
print("\nFULL TABLE: Education vs CRC")
display(pd.crosstab(case_control_df["education_clean"], case_control_df["CRC_status"], margins=True))

print("\nFULL TABLE: Healthcare Access vs CRC")
display(pd.crosstab(case_control_df["access_clean"], case_control_df["CRC_status"], margins=True))

print("\nFULL TABLE: Alcohol vs CRC")
display(pd.crosstab(case_control_df["alcohol_clean"], case_control_df["CRC_status"], margins=True))

# =========================================================
# Run Fisher tests for each category vs all others
# =========================================================
results = []

# Education
for cat in case_control_df["education_clean"].dropna().unique():
    if cat != "High school":   # reference group conceptually
        res = contingency_and_fisher(case_control_df, "education_clean", cat, outcome="CRC_status")
        results.append(res)

# Healthcare access
for cat in case_control_df["access_clean"].dropna().unique():
    if cat != "No barrier":    # reference group conceptually
        res = contingency_and_fisher(case_control_df, "access_clean", cat, outcome="CRC_status")
        results.append(res)

# Alcohol
for cat in case_control_df["alcohol_clean"].dropna().unique():
    if cat != "Never":         # reference group conceptually
        res = contingency_and_fisher(case_control_df, "alcohol_clean", cat, outcome="CRC_status")
        results.append(res)

# =========================================================
# Final results table
# =========================================================
fisher_results_df = pd.DataFrame(results).sort_values("p_value")

print("\nFINAL FISHER RESULTS TABLE")
display(fisher_results_df)

# optional: save results
fisher_results_df.to_csv("fisher_results_crc.csv", index=False)
print("Saved as fisher_results_crc.csv")

case_control_df.to_csv("case_control.csv", index=False)
