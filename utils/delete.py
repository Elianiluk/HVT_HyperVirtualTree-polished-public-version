import numpy as np
import matplotlib.pyplot as plt

# Target distribution (univariate normal)
def target_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Proposal distribution (normal with a certain standard deviation)
def proposal_distribution(x, std_dev=1.0):
    return np.random.normal(x, std_dev)

# Metropolis-Hastings algorithm
def metropolis_hastings(iterations, initial_state, proposal_std_dev):
    samples = [initial_state]
    current_state = initial_state

    for _ in range(iterations):
        # Propose a new state from the proposal distribution
        proposed_state = proposal_distribution(current_state, proposal_std_dev)

        # Calculate acceptance ratio
        acceptance_ratio = min(1, target_distribution(proposed_state) / target_distribution(current_state))

        # Accept or reject the proposed state
        if np.random.uniform(0, 1) < acceptance_ratio:
            current_state = proposed_state

        samples.append(current_state)

    return np.array(samples)

# Parameters
iterations = 5000
initial_state = 0.0
proposal_std_dev = 1.0

# Run Metropolis-Hastings algorithm
samples = metropolis_hastings(iterations, initial_state, proposal_std_dev)

# Plot the results
x_values = np.linspace(-4, 4, 1000)
plt.figure(figsize=(10, 6))

# Plot the target distribution
plt.plot(x_values, target_distribution(x_values), label='Target Distribution', color='red')

# Plot the samples from MCMC
plt.hist(samples, bins=50, density=True, label='MCMC Samples', alpha=0.5)

plt.title('Metropolis-Hastings MCMC Example')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()