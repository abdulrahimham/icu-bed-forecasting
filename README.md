# Forecasting ICU Bed Demand in California

Weekly forecasts of how many intensive care beds will be in use across California, 1 to 4 weeks ahead, built from public CDC hospital data.

## The Problem
An ICU (intensive care unit) is the part of a hospital for the sickest patients, such as people who need a machine to help them breathe. ICU beds are limited, because each one needs special equipment and specially trained nurses.

When many people get seriously ill at once, such as during a bad flu season or a COVID-19 wave, hospitals can run out of ICU beds. Fixing that takes time. Extra nurses have to be brought in, and planned surgeries that are not urgent may need to be delayed to free up space. If planners can see a surge coming a few weeks early, they have time to prepare.

## Who This Is For
Planners at the California Department of Public Health, the state agency that coordinates how California responds to health emergencies. The data covers the whole state, not individual hospitals, which matches the level where the department makes decisions.

## What the Forecast Does
- **Predicts:** the share of California's ICU beds in use, then converts it into a number of patients using recent bed counts. The share is forecast instead of the raw count because the count is distorted by changes in how many hospitals report, while the share stays steady (see Notebook 02).
- **How far ahead:** 1, 2, 3, and 4 weeks
- **Output:** a best guess plus a range the real number is likely to fall in (for example, "about 6,500 beds, likely between 6,200 and 6,900")

Different actions need different amounts of warning. Delaying surgeries can happen within a week, while bringing in extra staff takes longer. Forecasting several weeks ahead covers both, and measuring accuracy separately for each week shows how much to trust forecasts further out.

## How Success Is Measured
- **Error:** on average, how many beds the forecast was off by
- **Bar to beat:** a simple rule that says "next week will look like this week." This is hard to beat because ICU numbers usually change slowly. A model that cannot beat this rule is not worth using.
- **Honest ranges:** if the forecast says the real number has an 80% chance of falling within a range, it should actually land in that range about 80% of the time

The top of the range matters as much as the best guess. Running out of ICU beds is far more costly than having some sit empty, so planners need a realistic picture of the bad case.

## The General Problem
Outside healthcare, this is the same problem as forecasting demand when capacity is limited and slow to change:
- A retailer deciding how much stock to order before the holidays
- A delivery company planning how many drivers to schedule
- A tech company deciding how many servers to have ready before a traffic spike

In each case, guessing too low is costly, guessing too high wastes money, and adding capacity takes time.

## Data
[CDC National Healthcare Safety Network, Weekly Hospital Respiratory Data](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/ua7e-t2fy)

The CDC (Centers for Disease Control and Prevention) is the US federal public health agency. US hospitals report bed counts and patient numbers to it every week, and it publishes totals for each state. This project uses California's data, one row per week, from August 2020 to September 2026. The exact copy used is saved in `data/raw/` with the date it was downloaded, since the CDC sometimes updates past numbers.

## Known Limitations
- **Statewide only.** The forecast cannot say which hospitals or regions will be under the most pressure.
- **Reporting was optional from May through October 2024.** During that time, fewer hospitals sent in data, so lower totals may reflect missing reports rather than fewer patients.
- **The COVID-19 years.** 2020 to 2022 include surges far larger than anything since, which may not reflect what future years look like.

## Key Terms
| Term | Meaning |
|---|---|
| ICU | Intensive care unit, the hospital area for the most seriously ill patients |
| Staffed bed | A bed with the equipment and nurses needed to actually care for a patient |
| Occupancy | How many beds have a patient in them |
| Forecast horizon | How far ahead a forecast looks (here, 1 to 4 weeks) |
| Baseline | A simple rule that any useful model must beat |
| COVID-19, flu, RSV | Three common illnesses that affect breathing and drive winter ICU demand |

## Project Structure
```
data/raw/          Original downloaded data, never edited
data/processed/    Cleaned data created by the notebooks
notebooks/         Analysis, numbered in order
src/               Reusable code
```

## How to Run
```
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Then run the notebooks in order.

## Status
In progress.
- [x] 01: Data download and data quality checks
- [x] 02: Choosing the forecasting target and flagging unreliable weeks
- [x] 03: Exploratory analysis (seasonal pattern, volatility, what drives the winter rise, bed capacity)
- [ ] 04: Baseline forecasts
- [ ] 05: Forecasting models
- [ ] 06: Evaluation