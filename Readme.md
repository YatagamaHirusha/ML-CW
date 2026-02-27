# Project MatchMind: A Deep Contextual Bandit Approach to Psychological Matchmaking

**Student Name:** Hirusha  
**Module:** [Insert Module Name / Code]  
**Date:** [Insert Current Date]  

---

## 1. Abstract
Modern dating applications rely heavily on shallow collaborative filtering or simple interest-based matching, leading to high churn rates and psychologically incompatible pairings. This project introduces **MatchMind**, an AI-driven matchmaking engine that optimizes for long-term relational stability using the ECR-RS (Experiences in Close Relationships-Revised) attachment scale. The system was developed using a two-phase machine learning pipeline. First, a Deep Neural Surrogate Model was trained to approximate complex psychological heuristics. Recognizing the limitations of supervised approximations, the architecture was evolved into a **Deep Q-Network (DQN) Contextual Bandit**. By interacting with a stochastic user simulation environment, the DQN utilized an Epsilon-Greedy policy and Experience Replay to learn matching policies strictly from binary feedback (Confirm/Skip). The final model achieved a **92.8% Confirm Rate** in simulation and an **81.0% generalization rate** on unseen users during 5-Fold Cross-Validation, proving high stability and sample efficiency.

## 2. Introduction & Motivation
The primary challenge in algorithmic matchmaking is the "Anxious-Avoidant Trap"—a well-documented psychological phenomenon where anxiously attached individuals repeatedly pair with dismissive-avoidant individuals, resulting in volatile relationships. The objective of MatchMind is to move beyond superficial hobby overlaps and build a recommendation system that penalizes toxic psychometric pairings while rewarding secure attachment and aligned relationship intents. 

## 3. Dataset Characteristics & Profiling
The foundational dataset consists of 1,200 uniquely generated user profiles, rigorously cleaned to remove null values and irrelevant identifiers. The data represents a highly complex, multi-modal feature space combining continuous psychometrics, strict categorical logistics, and multi-label text arrays.

The dataset is structured into three primary domains:
1. **Demographic Logistics (Categorical):** Age, Gender, Target Gender, and Geographic Location. These serve as absolute constraints.
2. **Psychometric ECR-RS Scores (Continuous):** `Anxiety` and `Avoidance` scores, evaluated on a standard psychological scale of 1.0 to 7.0. 
3. **Multi-Label Categorical Vectors (JSON Arrays):** 11 distinct preference categories (e.g., *Relationship Intent, Lifestyle, Movies & Shows, Gaming & Digital*). Each user possesses an array of string values representing their specific interests.

## 4. Data Preprocessing, Normalization & Feature Engineering
Neural networks require bounded, standardized numerical inputs to prevent exploding gradients and ensure uniform weight distribution. A critical phase of this project was transforming the multi-modal dataset into a normalized **16-Dimensional State Vector Tensor** ($X \in \mathbb{R}^{16}$).

### 4.1. Stage 1: Deterministic Database Filtering ($O(1)$ Complexity)
To optimize inference speed and prevent the Neural Network from wasting tensor parameters on basic SQL-style logic, absolute constraints (Gender orientation matching and Location matching) were processed as strict database queries prior to model invocation. This ensures the Neural Network only evaluates logically viable candidate pools.

### 4.2. Stage 2: Min-Max Normalization of Continuous Variables
To map continuous variables into standard $[0.0, 1.0]$ bounds, targeted normalization equations were applied:
* **Age Gap Normalization:** Instead of feeding raw ages, the absolute age difference was calculated and normalized by a factor of 10.0 (assuming a standard acceptable age gap variance). 
  $$X_{age\_gap} = \frac{|Age_A - Age_B|}{10.0}$$
* **Psychometric Normalization:** The ECR-RS anxiety and avoidance traits max out at 7.0. They were min-max scaled to the network's preferred range:
  $$X_{anx} = \frac{Anxiety}{7.0} \quad \text{and} \quad X_{avo} = \frac{Avoidance}{7.0}$$

### 4.3. Stage 3: Feature Extraction via Jaccard Similarity

The 11 multi-label text arrays (e.g., lists of hobbies) could not be directly passed into a linear layer. Instead of sparse one-hot encoding, which would cause the curse of dimensionality, we extracted the mathematical overlap between two users using the **Jaccard Similarity Coefficient**. For any two arrays of interests $A$ and $B$:
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
This elegantly compresses complex string arrays into 11 distinct, continuous $[0.0, 1.0]$ features, allowing the network to easily weigh the importance of shared "Relationship Intent" versus shared "Gaming" preferences.

## 5. Methodological Justification
Selecting the correct machine learning paradigm was critical for this recommendation engine. 
* **Why Unsupervised Learning is Insufficient:** While clustering algorithms ($K$-Means) or dimensionality reduction are excellent for segmenting users into "dating personas," they cannot *optimize* a pairing policy. Clustering dictates who is identical, but compatible dating often requires complementary traits.
* **Why Pure Supervised Learning is Insufficient:** Supervised learning requires a massive, labeled dataset of "perfect couples." Training a network to merely copy our handcrafted rule-engine creates a circular dependency where the AI adds no novel intelligence; it mathematically cannot outperform the formula it is imitating.
* **The Reinforcement Learning Solution:** A **Contextual Bandit** (a specialized class of RL where the episode length is 1) is the optimal paradigm. It allows the agent to observe a Context (the 16-D normalized State Vector), take an Action (recommend or skip), and learn policies directly from environmental Rewards.

## 6. Model Architecture: The Two-Phase Approach
The project was developed in two distinct architectural phases to demonstrate the evolution from rule-based approximation to true stochastic learning.

### 6.1. Approach A: Deep Neural Surrogate Regressor
Initially, a deterministic heuristic engine was built to mathematically score pairs based on weighted categories and psychometric penalties. A multi-layer perceptron was trained via Mean Squared Error (MSE) to predict this exact score. While accurate, it acted purely as a surrogate function, prompting the upgrade to Phase 2.

### 6.2. Approach B: True Contextual Bandit DQN (Final Model)
To break the circular dependency, the system was upgraded to a true Contextual Bandit decoupled from the underlying heuristic math.

* **Network Topology:** A deep neural network mapping $16 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 1$, outputting a single Expected Reward. Dropout layers ($p=0.2$) were included to prevent overfitting.
* **Experience Replay:** To maximize sample efficiency, an Experience Replay buffer (`maxlen=10000`) was introduced. The agent stored `(State, Reward)` tuples and sampled mini-batches of 64 memories for gradient updates, breaking temporal correlations in the training data.
* **Exploration vs. Exploitation:** An Epsilon-Greedy policy was implemented. $\epsilon$ was initialized at 1.0 (100% random exploration) and slowly decayed by a factor of 0.999 down to 0.05, allowing the agent to dynamically explore the environment before committing to an exploitation strategy.
* **Optimization:** The Adam optimizer ($\alpha = 0.001$) was utilized alongside a `StepLR` learning rate scheduler ($\gamma = 0.5$) to fine-tune convergence in the later epochs.

## 7. The Stochastic Environment Simulator
Because the system lacks live user telemetry (cold-start), an offline Stochastic Simulator was engineered to act as the environment. 
When the Agent recommends a candidate, the simulator evaluates the underlying psychological compatibility, adds probabilistic noise, and returns a binary outcome:
* **CONFIRM (+1.0 Reward):** The simulated user accepted the match.
* **SKIP (-1.0 Reward):** The simulated user rejected the match.

The maximum probability of a Confirm was capped at 95% to simulate the inherent unpredictability and noise of human UI interaction.

## 8. Experimental Evaluation & Results
The Contextual Bandit was evaluated against two baselines over 500 simulated sessions:
1. **Random Guesser:** Recommends a random viable candidate.
2. **Hobby Baseline:** Greedily recommends the candidate with the highest Jaccard similarity in interests, ignoring psychology.

### 8.1. Baseline Simulation Results
* **DRL Agent:** 92.8% Confirm Rate
* **Hobby Baseline:** 82.4% Confirm Rate
* **Random Guesser:** 53.6% Confirm Rate

The DQN successfully learned to prioritize "Relationship Intent" and penalize "Anxious-Avoidant" pairings without explicit programming, outperforming the greedy hobby baseline by over 10%.

### 8.2. 5-Fold User-Level Cross-Validation
To mathematically prove the model learned universal psychological rules rather than memorizing specific users, a rigorous 5-Fold Environment Split was conducted. The environment was partitioned; independent agents were trained on 80% of the users and evaluated purely on 20% *unseen* users, with total memory wiping between folds to prevent data leakage.
* **Average Generalization Rate:** 81.0%
* **Standard Deviation (Stability):** $\pm 2.0\%$

The generalization gap (drop from 92.8% to 81.0%) proves the evaluation framework was rigorous. The extremely tight standard deviation ($\pm 2.0\%$) confirms the DQN policy is highly robust and stable across diverse demographic splits.

## 9. Conclusion
Project MatchMind successfully demonstrates the application of Deep Reinforcement Learning in complex sociometric environments. By applying mathematically rigorous normalization and Jaccard feature extraction, the system successfully digested multi-modal human profiles. Furthermore, pivoting from a Surrogate Regressor to a Contextual Bandit with Experience Replay allowed the Neural Network to organically discover and optimize for non-linear psychological stability, paving the way for scalable matchmaking ecosystems.