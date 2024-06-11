
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from reads.tsv
reads = pd.read_table('reads.tsv')

np.random.seed(123)

# Define prior probabilities
prior_prob = {
    "AA": 0.95**2,
    "TT": 0.05**2,
    "AT": 1 - (0.95**2 + 0.05**2)
}

def log_likelihood(observation, probability_of_error, genotype):
    
    #Calculate the log likelihood of observing a specific genotype given the observation
    #and probability of error for that observation.

    if genotype == "AA":
        if observation == 'A':
            return np.log(1 - probability_of_error)
        elif observation == 'T':
            return np.log(probability_of_error)
    elif genotype == "TT":
        if observation == 'T':
            return np.log(1 - probability_of_error)
        elif observation == 'A':
            return np.log(probability_of_error)
    elif genotype == "AT":
        return np.log(0.5) 


def calculate_genotype_log_likelihoods(df):

   # Calculate log likelihoods for each genotype

    AA_ll = 0
    TT_ll = 0
    AT_ll = 0

    for index, row in df.iterrows():
        observation = row['observations']
        probability_of_error = row['probability_of_error']

        AA_ll += log_likelihood(observation, probability_of_error, "AA")
        TT_ll += log_likelihood(observation, probability_of_error, "TT")
        AT_ll += log_likelihood(observation, probability_of_error, "AT")

    return AA_ll, TT_ll, AT_ll


AA_postprobs = []
AT_postprobs = []
TT_postprobs = []


iterations = 1000

for i in range(iterations):
    random_sample = reads.sample(n=50, replace=True)

    # Calculate log likelihoods for observed data
    AA_LogLike, TT_LogLike, AT_LogLike = calculate_genotype_log_likelihoods(random_sample)


    # Calculate observed log probability
    observed_logprob = np.log(np.exp(AA_LogLike + np.log(prior_prob["AA"])) +
                            np.exp(TT_LogLike + np.log(prior_prob["TT"])) +
                            np.exp(AT_LogLike + np.log(prior_prob["AT"])))

    # Calculate posterior probabilities using log likelihoods and prior probabilities
    AA_postprob = np.exp(AA_LogLike + np.log(prior_prob["AA"]) - observed_logprob)
    AT_postprob = np.exp(AT_LogLike + np.log(prior_prob["AT"]) - observed_logprob)
    TT_postprob = np.exp(TT_LogLike + np.log(prior_prob["TT"]) - observed_logprob)

    AA_postprobs.append(AA_postprob)
    AT_postprobs.append(AT_postprob)
    TT_postprobs.append(TT_postprob)

mean_AA = np.mean(AA_postprobs)
std_AA = np.std(AA_postprobs)

# Compute mean and standard deviation for AT_postprobs
mean_AT = np.mean(AT_postprobs)
std_AT = np.std(AT_postprobs)

# Compute mean and standard deviation for TT_postprobs
mean_TT = np.mean(TT_postprobs)
std_TT = np.std(TT_postprobs)

# Print mean and standard deviation with descriptive labels
print(f"Mean Posterior Probability for Genotype AA: {mean_AA:.4f}")
print(f"Standard Deviation for Genotype AA: {std_AA:.4f}")

print(f"Mean Posterior Probability for Genotype AT: {mean_AT:.4f}")
print(f"Standard Deviation for Genotype AT: {std_AT:.4f}")

print(f"Mean Posterior Probability for Genotype TT: {mean_TT:.4f}")
print(f"Standard Deviation for Genotype TT: {std_TT:.4f}")

# Plot histograms using seaborn
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(AA_postprobs, bins=400, kde=True)
plt.title('Posterior Probability for Genotype AA')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.xlim(0,0.1)


plt.subplot(1, 3, 2)
sns.histplot(AT_postprobs, bins=400, kde=True)
plt.title('Posterior Probability for Genotype AT')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.xlim(0.9,1)

plt.subplot(1, 3, 3)
sns.histplot(TT_postprobs, bins=400, kde=True)
plt.title('Posterior Probability for Genotype TT')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.xlim(0,0.0001)



plt.tight_layout()
plt.show()

