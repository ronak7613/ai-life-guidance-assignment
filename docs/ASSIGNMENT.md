# AI Life Guidance Engine - Take Home Assignment

## 1. Background

We are building an AI based life guidance assistant that feels similar to an astrology product but does not use any real astrology rules.

Instead, we will use:

* Basic user profile traits
* Simple life events
* A synthetic "life archetype" rule book

Your task is to design a small but realistic end to end system that can take a user and their events, and output structured guidance suggestions.

This assignment is scoped for about 10 to 12 hours of work.

---

## 2. Data provided

Under `data/` you will find:

* `users.csv` - synthetic user profiles
* `events.csv` - recent events for each user
* `guidance_rules.csv` - a simple rule set for life archetypes and recommended guidance

You are free to create additional derived CSVs or JSON files if that helps your approach.

---

## 3. What you need to build

### 3.1 Data understanding and prep

* Load all CSV files from `data/`
* Do basic sanity checks on distribution, missing values, and label balance
* Create a small EDA notebook at `notebooks/01_eda.ipynb` that shows:

  * Key descriptive stats
  * Any new features you decide to derive
  * Short textual notes on your observations

### 3.2 Scoring and rule engine

Using `guidance_rules.csv`, design a rule or scoring engine that:

* Assigns each user into one or more "life archetypes" based on profile and events
* Produces 2 to 4 guidance suggestions per user in a structured JSON like format:

  * `category` (career, health, relationships, mindset)
  * `priority` (1 to 5)
  * `message` (1 or 2 short sentences)
* Implement this as reusable Python code in `src/` (for example `src/model_pipeline.py`)

### 3.3 Simple ML or ranking component

Add one ML component that improves or personalises the guidance. Examples:

* A classifier that predicts high stress users and boosts health guidance
* A ranking model that orders guidance suggestions based on engagement labels that you simulate
* Any other small supervised model you design from this synthetic data

You do not need a perfect model. Focus on clear pipeline, feature design, and evaluation.

### 3.4 Inference interface

Provide one simple way to run the system end to end for a single user id:

Option A - CLI script

* `python src/app.py --user_id 12`
* Prints the final structured guidance for that user

Option B - Notebook

* A notebook `notebooks/02_inference_demo.ipynb` that calls your pipeline and displays guidance for 3 to 5 example users

### 3.5 Evaluation and logging

Add a basic evaluation layer:

* Define at least one metric that is meaningful for this synthetic task
* Log:

  * Few example inputs and outputs
  * Distribution of guidance categories and priorities
  * At least one simple chart or table

Put short commentary into `docs/RESULTS.md` that explains:

* What you tried
* What worked and what did not
* What you would do next with more time

---

## 4. Technical expectations

We are looking for:

* Clean, readable Python code
* Sensible folder structure
* Separation between data loading, feature engineering, model logic, and interface
* Use of virtualenv or conda environment is welcome but not required

A simple `requirements.txt` should list the main dependencies, for example:

* pandas
* numpy
* scikit-learn
* jupyter
* matplotlib or seaborn (optional)

You are free to use additional open source libraries if you mention them.

---

## 5. What to submit

When you send the assignment back, the repo should contain:

* `README.md`
* `docs/ASSIGNMENT.md` and `docs/RESULTS.md`
* `data/*.csv` (can be the same synthetic data we provided and any new files you add)
* `src/` Python modules as described
* `notebooks/01_eda.ipynb` and optionally `02_inference_demo.ipynb`
* `requirements.txt`

We should be able to:

1. Clone the repo
2. Install dependencies from `requirements.txt`
3. Run your pipeline according to your README instructions
4. Inspect your notebooks and code to understand your thinking

Focus on clarity and structure more than complicated models.
