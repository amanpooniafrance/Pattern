# PROJECT SPECIFICATION

## Title

Supervised State Aggregation for Markov-Based EURUSD State Prediction

---

# Objective

Build a complete research framework that:

1. Downloads EURUSD historical data.
2. Converts returns into discrete states.
3. Constructs M1–M5 Markov state models.
4. Estimates transition probabilities.
5. Aggregates similar states using multiple methods.
6. Re-estimates transition probabilities after aggregation.
7. Compares clustering and tree-based approaches.
8. Produces visualizations, summary statistics, and exportable results.

The code should be fully reproducible from raw market data without requiring any pre-existing objects.

---

# DATA ACQUISITION

Download daily EURUSD data using yfinance.

Ticker:

EURUSD=X

Date range:

2003-12-01 to latest available date

Required columns:

* Date
* Close

---

# DATA PREPARATION

## Step 1

Calculate daily log returns:

log_ret(t) = ln(Close(t) / Close(t-1))

---

## Step 2

Calculate quantiles:

30th percentile
70th percentile

---

## Step 3

Convert returns into discrete states:

State = -1 if return < q30

State = 0 if q30 <= return <= q70

State = +1 if return > q70

---

# STATE CONSTRUCTION

Support:

* M1
* M2
* M3
* M4
* M5

User should be able to select:

LOOKBACK = 5

For M5:

Features:

s1 = state(t-5)

s2 = state(t-4)

s3 = state(t-3)

s4 = state(t-2)

s5 = state(t-1)

Target:

Y = state(t)

---

# TRANSITION TABLE

For every state:

Estimate:

P(Y=-1)

P(Y=0)

P(Y=+1)

using observed frequencies.

Store:

transition_counts

transition_probs

state_counts

state_probability

---

# BAYESIAN SHRINKAGE

Before clustering:

Apply Bayesian smoothing.

Use:

alpha = 5

Compute global probabilities.

Shrink rare-state estimates toward the global distribution.

Store:

transition_probs_smoothed

Use smoothed probabilities for clustering.

Use raw counts for final re-estimation.

---

# STATE REPRESENTATION

Represent every state by:

V = [P(-1), P(0), P(+1)]

using smoothed probabilities.

---

# DERIVED FEATURES

For every state calculate:

## Direction

Direction = P(+1) - P(-1)

Interpretation:

Negative = Bearish

Positive = Bullish

Near zero = Neutral

---

## Entropy

Entropy = -Σ p log(p)

Measure of uncertainty.

Low entropy = strong signal.

High entropy = weak signal.

---

## Gini

Gini = 1 - Σ p²

Measure of impurity.

---

## Classification Error

Classification_Error = 1 - max(P)

Measure of misclassification probability.

---

## Confidence

Confidence = max(P)

Measure of strongest predicted outcome.

---

# STATE AGGREGATION METHODS

Implement and compare four methods.

---

# METHOD 1

JENSEN-SHANNON CLUSTERING

Representation:

[P(-1), P(0), P(+1)]

Distance:

Jensen-Shannon Distance

Implementation:

scipy.spatial.distance.jensenshannon

Clustering:

Agglomerative Hierarchical Clustering

Linkage:

average

Do NOT use Ward linkage.

Determine optimal cluster count automatically.

Evaluate:

k = 2 through 20

using silhouette score.

Select best k.

Output:

state_to_cluster_js

---

# METHOD 2

SSE / EUCLIDEAN CLUSTERING

Representation:

[P(-1), P(0), P(+1)]

Distance:

Euclidean Distance

Equivalent to SSE minimization.

Clustering:

Agglomerative Hierarchical Clustering

Linkage:

average

Determine optimal cluster count automatically.

Evaluate:

k = 2 through 20

using silhouette score.

Select best k.

Output:

state_to_cluster_sse

---

# METHOD 3

ENTROPY TREE WITH DIRECTION

Purpose:

Create supervised state aggregation using a decision tree.

Features:

Direction

Entropy

Target:

Dominant future state

defined as:

argmax(P(-1), P(0), P(+1))

Tree:

DecisionTreeClassifier

criterion='entropy'

Tune:

max_depth = 1 through 10

using cross-validation.

Choose best depth.

Leaf nodes become clusters.

Output:

state_to_cluster_entropy

---

# METHOD 4

GINI TREE WITH DIRECTION

Features:

Direction

Gini

Target:

Dominant future state

Tree:

DecisionTreeClassifier

criterion='gini'

Tune:

max_depth = 1 through 10

using cross-validation.

Choose best depth.

Leaf nodes become clusters.

Output:

state_to_cluster_gini

---

# IMPORTANT NOTE

Entropy, Gini and Classification Error are NOT distance metrics.

They are impurity measures.

Direction must be included in tree features.

Without Direction:

[0.60,0.20,0.20]

and

[0.20,0.20,0.60]

have identical entropy and identical gini despite opposite market direction.

Direction resolves this issue.

---

# CLUSTER RE-ESTIMATION

For every method:

Do NOT average probabilities.

Instead:

Pool raw transition counts.

Example:

Cluster contains:

State A

State B

State C

Cluster counts:

Count(A) + Count(B) + Count(C)

Recompute:

P(-1)

P(0)

P(+1)

from pooled counts.

This produces statistically valid cluster transition probabilities.

---

# CLUSTER ORDERING

All cluster outputs must be sorted by:

Direction = P(+1) - P(-1)

Order:

Most Bearish

Bearish

Neutral

Bullish

Most Bullish

This ordering must be used everywhere:

Tables

Plots

Exports

Heatmaps

CSV files

---

# CLUSTER SUMMARY TABLE

For every method create:

Cluster_ID

N_States

N_Observations

P(-1)

P(0)

P(+1)

Direction

Entropy

Gini

Classification_Error

Confidence

Sorted by Direction.

---

# CLUSTER MEMBERS TABLE

For every cluster create:

Cluster

State

State_Count

P(-1)

P(0)

P(+1)

Direction

Entropy

Gini

Classification_Error

Sorted according to original state ordering.

---

# VISUALIZATIONS

Generate:

## 1. Transition Probability Heatmap

Rows:

Clusters

Columns:

-1

0

+1

---

## 2. Cluster Size Distribution

Number of observations per cluster.

---

## 3. Direction Distribution

Direction score by cluster.

---

## 4. Dendrogram

For:

JS clustering

SSE clustering

---

## 5. Decision Tree Diagram

For:

Entropy Tree

Gini Tree

Display:

Split variables

Thresholds

Leaf distributions

Cluster IDs

---

# COMPARISON REPORT

Create one final comparison table:

Method

Number_of_Clusters

Average_Cluster_Size

Average_Entropy

Average_Confidence

Average_Direction

Silhouette_Score

Cross_Validation_Score

where applicable.

---

# EXPORTS

Export all outputs to CSV.

Required files:

state_transition_table.csv

js_cluster_summary.csv

sse_cluster_summary.csv

entropy_tree_summary.csv

gini_tree_summary.csv

cluster_members_js.csv

cluster_members_sse.csv

cluster_members_entropy.csv

cluster_members_gini.csv

comparison_report.csv

---

# CODE REQUIREMENTS

Use:

numpy

pandas

scipy

scikit-learn

matplotlib

seaborn

yfinance

Code must be:

* fully runnable
* heavily commented
* modular
* reproducible
* suitable for academic research

Every major section must contain comments explaining:

1. Mathematical objective.
2. Statistical interpretation.
3. Why the step is performed.
4. Expected output.

The script should run from raw EURUSD download through final exported reports with no manual intervention.
