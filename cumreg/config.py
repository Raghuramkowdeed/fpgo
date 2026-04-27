"""Configuration dataclass for cumulative regret experiments."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Dataset
    dataset: str = "livecodebench"  # "livecodebench", "gsm8k"
    data_dir: str = "data"
    seed: int = 42

    # Mode
    mode: str = "single_turn"  # "single_turn" or "multi_turn"
    max_turns: int = 3  # for multi-turn: max repair attempts

    # Model
    model_name: str = "allenai/OLMo-3-7B-Instruct"
    load_in_4bit: bool = True
    max_new_tokens: int = 2048
    max_seq_length: int = 8192

    # Generation
    do_sample: bool = False  # greedy by default
    temperature: float = 0.0
    top_p: float = 1.0

    # Best-of-N generation
    n_generations: int = 5  # candidates per problem (1 = single greedy)
    cache_temperature: float = 0.7  # sampling temp for candidate generation

    # Retrieval
    retrieval_strategy: str = "knn"  # "knn", "recency", "diversity"
    k_shots: int = 3
    embedding_model: str = "all-MiniLM-L6-v2"
    max_history_batches: Optional[int] = None  # None = unbounded
    retrieve_correct_only: bool = True  # only use passing solutions
    min_reward_threshold: float = 0.5  # min score to retrieve as ICL example

    # OLMo embedding reuse (instead of SentenceTransformer)
    use_olmo_embeddings: bool = True  # reuse loaded model for embeddings
    embedding_layer: int = -1  # which hidden layer (-1 = last)
    embedding_max_length: int = 512  # max tokens for embedding forward pass

    # Experiment
    batch_size: int = 1
    num_steps: Optional[int] = None  # None = exhaust dataset
    exp_dir: str = "results"
    checkpoint_every: int = 50

    # Code execution
    execution_timeout: int = 10  # seconds per test case
    use_private_tests: bool = False
