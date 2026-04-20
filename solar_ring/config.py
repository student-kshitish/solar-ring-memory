"""All hyperparameters for Solar Ring Memory architecture."""

# Model dimensions
D_MODEL = 300       # matches GloVe 300d
N_LAYERS = 8
VOCAB_SIZE = 32000  # will be overridden by tokenizer

# Ring structure
MAX_RINGS = 13       # 1 sun + 4 planets + 8 moons
SLOTS_PER_RING = 8  # SUBJ(0), OBJ(1), VERB(2), ROT0..4(3..7)
ROTATING_SLOTS = 5  # slots 3-7

# Ring slot indices
SUBJ_SLOT = 0
OBJ_SLOT = 1
VERB_SLOT = 2
ROT_START = 3       # rotating buffer starts here

# POS role IDs
ROLE_SUBJ = 1
ROLE_OBJ = 2
ROLE_VERB = 3
ROLE_PREP = 4
ROLE_CONJ = 5
ROLE_ADJ = 6
ROLE_DET = 7
ROLE_OTHER = 8
NUM_ROLES = 9       # 0-indexed background + 8 roles

# Conjunction triggers (dep labels that spawn child rings)
SPAWN_CONJUNCTIONS = {"mark", "relcl", "advcl", "ccomp", "xcomp", "acl"}

# Flatten output size
FLAT_SIZE = MAX_RINGS * SLOTS_PER_RING * D_MODEL  # 13*8*300 = 31200

# Training
LR = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
BATCH_SIZE = 4
MAX_SEQ_LEN = 128

# Loss weights
LAMBDA_POS = 0.3
LAMBDA_SPAWN = 0.2
LAMBDA_RESOLVE = 0.2

# Cross-ring attention: only layer index 4 (0-indexed, layer 5)
CROSS_RING_LAYER = 4
# Pronoun resolution: layer 6
PRONOUN_LAYER = 5
# Relation encoder: layer 7
RELATION_LAYER = 6

# ── Enhanced pronoun resolution ──────────────────────────────────────────
# Recency decay base: 0.7^dist penalises distant antecedents
PRONOUN_RECENCY_DECAY = 0.7
# Confidence boost multiplier for locked (write-once) slots
PRONOUN_CONF_SUBJ = 1.5   # SUBJ locked boost
PRONOUN_CONF_OBJ  = 1.3   # OBJ locked boost
# OBJ-vs-SUBJ relative prior (slight preference for subject antecedents)
PRONOUN_OBJ_PRIOR = 0.8
# Sun State prior weight (global context, used as fallback)
PRONOUN_SUN_PRIOR = 0.3

# ── Contrastive loss ────────────────────────────────────────────────────
LAMBDA_CONTRASTIVE = 0.15   # weight for InfoNCE pronoun contrastive loss
CONTRASTIVE_TEMP   = 0.07   # temperature for contrastive softmax

# ── Multi-hop relation encoder ──────────────────────────────────────────
LAMBDA_RELATION = 0.1       # weight for relation consistency loss
