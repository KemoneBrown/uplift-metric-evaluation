"""Central configuration: seeds, partition sizes and decision thresholds.

Every value the analysis depends on lives here, recorded in version
control and traceable to the OSF pre-registration (10.17605/OSF.IO/5AYKJ).
"""

# --- Reproducibility (NFR1) ---
RANDOM_SEED = 42

# --- Partition sizes (OSF Sampling Plan 3.3, 3.4) ---
TRAIN_SIZE = 1_000_000
TEST_SIZE = 1_000_000

# --- Subsampling (OSF 2.3, 3.4) ---
N_SUBSAMPLES = 50
SUBSAMPLE_SIZE = 100_000
N_SUBSAMPLES_CONTINGENCY = 100

# --- Stratification keys (OSF 3.3) ---
STRATIFY_COLS = ("treatment", "conversion")

# --- Decision rule (OSF 5.3) ---
TAU_THRESHOLD = 0.7
BOOTSTRAP_RESAMPLES = 10_000
CI_LEVEL = 0.95
H2A_ALPHA = 0.05

# --- Models under comparison (OSF 2.3) ---
MODELS = ("s_learner", "t_learner", "kl_tree", "no_information_baseline")