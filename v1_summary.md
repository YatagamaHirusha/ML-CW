# V1 Summary Report

## Overview

`v1.ipynb` is the first version of the ML coursework project. Its goal is to build a **psychology-driven dating match recommendation system**. This version focuses on two key tasks:

1. **Synthetic dataset generation** — creating a realistic pool of 600 user profiles.
2. **Compatibility algorithm** — scoring and ranking potential matches for a given user.

---

## What We Did

### 1. Dataset Generation

We used Python's `Faker` library (for names) alongside `NumPy` and the built-in `random` module to procedurally generate 600 user profiles. Each profile contains the following fields:

| Field | Description |
|---|---|
| `user_id` | Unique integer identifier (1–600) |
| `name` | Randomly generated first name |
| `gender` | `Male` or `Female`, chosen at random |
| `age` | Random integer between 18 and 35 |
| `target_gender` | Opposite of the user's own gender |
| `anxiety_score` | ECR-RS attachment anxiety score (1.0–7.0) |
| `avoidance_score` | ECR-RS attachment avoidance score (1.0–7.0) |
| `attachment_style` | Derived from anxiety/avoidance scores (see below) |
| `interests` | 3 randomly sampled interests from a pool of 15 |

**Interest pool:** Hiking, Gaming, Travelling, Reading, Cooking, Music, Art, Movies, Swimming, Meditation, Coding, Drink, Shopping, Badminton, Fitness.

### 2. Attachment Style Classification (ECR-RS)

Attachment scores are generated using a **bimodal normal distribution** to reflect real-world clustering:

- 50% of users belong to a **secure cluster** — scores drawn from `Normal(μ=2.5, σ=1.0)` for both anxiety and avoidance.
- 50% of users belong to an **insecure cluster** — scores drawn from `Normal(μ=4.5, σ=1.5)`, with higher variance.

All scores are clipped to the valid ECR-RS range of **1.0 to 7.0**. A midpoint of **4.0** is used as the threshold to classify each user into one of four attachment styles:

| Attachment Style | Anxiety | Avoidance |
|---|---|---|
| Secure | < 4.0 | < 4.0 |
| Anxious-Preoccupied | ≥ 4.0 | < 4.0 |
| Dismissive-Avoidant | < 4.0 | ≥ 4.0 |
| Fearful-Avoidant | ≥ 4.0 | ≥ 4.0 |

### 3. Compatibility Algorithm

The `calculate_compatibility(user_a, user_b)` function scores how well two users match, returning a value between 0 and 100.

**Step 1 — Hard Filters (instant disqualification):**
- Gender preference must be satisfied (`target_gender` of A must equal `gender` of B).
- Age difference must be ≤ 5 years.

If either filter fails, the function returns `0`.

**Step 2 — Psychological Compatibility Score (80% weight):**

The score rewards pairs who are collectively closer to the secure (low anxiety, low avoidance) end of the spectrum:

```
pair_anxiety   = (user_a.anxiety_score + user_b.anxiety_score) / 2
pair_avoidance = (user_a.avoidance_score + user_b.avoidance_score) / 2
stability_penalty = (pair_anxiety + pair_avoidance) / 14.0   # max combined sum is 14
psych_score = 1.0 - stability_penalty
```

**Step 3 — Interest Overlap Score (20% weight):**

Each shared interest adds 0.1 to the interest score (maximum 0.3 for 3 shared interests):

```
interest_score = |{user_a.interests} ∩ {user_b.interests}| × 0.1
```

**Final Score:**

```
final_score = (psych_score × 0.8 + interest_score × 0.2) × 100
```

---

## Dataset Statistics

After generating all 600 profiles (`df.describe()`):

| Metric | Age | Anxiety Score | Avoidance Score |
|---|---|---|---|
| Mean | 26.37 | 3.58 | 3.52 |
| Std Dev | 5.11 | 1.53 | 1.56 |
| Min | 18 | 1.00 | 1.00 |
| Max | 35 | 7.00 | 7.00 |

---

## Sample Match Results

A test run was performed for user **Jacqueline** (Dismissive-Avoidant, Anxiety: 2.81, Avoidance: 4.32, Interests: Music, Art, Badminton):

| Rank | Name | Attachment Style | Score | Shared Interests |
|---|---|---|---|---|
| 1 | Melissa | Secure | 55.91% | Music |
| 2 | Megan | Secure | 55.09% | Badminton |
| 3 | Pamela | Secure | 54.23% | Music |
| 4 | Victoria | Secure | 53.17% | Art |
| 5 | John | Secure | 53.11% | Art, Music |

**Key observation:** All top matches have a **Secure** attachment style. This is expected because Jacqueline's scores sit mostly in the lower range (low anxiety), and the algorithm rewards couples whose combined anxiety and avoidance averages are low, naturally favouring Secure partners.

---

## Limitations & Next Steps

- The dataset is entirely **synthetic** — real-world attachment scores and user behaviour may differ significantly.
- Gender preferences are currently hard-coded as heterosexual only (opposite-gender targeting).
- The compatibility formula uses a simple linear penalty; future versions could incorporate more nuanced attachment theory models (e.g., rewarding complementary styles).
- Version 2 could introduce machine learning models trained on interaction data to improve match quality beyond the rule-based algorithm used here.
