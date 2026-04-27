"""
FGPO (Frontier-Guided Policy Optimization) -- Pseudocode

Notation
--------
    x         : a coding problem (statement + hidden test cases)
    R(y, x)   : oracle reward in [0, 1] = fraction of hidden tests passed by code y
    pi_theta  : policy being trained
    pi_ref    : frozen reference policy (for the KL term)
    F         : frontier model used as a hint generator
"""

from math import inf
from typing import Optional


# =====================================================================
# Algorithm 1 -- Step-1 Frontier Loop  (run once per problem)
# =====================================================================
def step1_loop(
    x,                  # one problem
    pi_theta,           # current policy
    F,                  # frontier model used to propose hints
    R,                  # oracle reward function
    N: int,             # max refinement rounds
    K: int,             # probe samples per round
    tau: float,         # early-stop reward threshold
    worst_k: int,       # # worst samples shown to F
    best_k: int,        # # best  samples shown to F
):
    history = []                      # round summaries fed back to F
    best_hint, best_avg = None, -inf  # best hinted round seen so far
    h: Optional[str] = None           # current hint; None at round 0

    for r in range(N):
        # (a) probe pi_theta with the current hint
        prompt  = build_prompt(x.question, h)            # prepends "Hint: ..." iff h is not None
        samples = pi_theta.sample(prompt, n=K, temperature=1.0)
        rewards = [R(y, x) for y in samples]
        avg_r   = mean(rewards)

        # (b) record the best hinted round (round 0 baseline never qualifies)
        if h is not None and avg_r > best_avg:
            best_hint, best_avg = h, avg_r

        # (c) early stop if the current hint already solves x
        if h is not None and avg_r >= tau:
            break

        # (d) summarize this round for F: worst_k + best_k samples + their oracle errors
        shown = pick_worst_and_best(samples, rewards, worst_k, best_k)
        history.append({"hint": h, "avg_reward": avg_r, "shown_samples": shown})

        # (e) ask F for a refined hint; keep previous hint on failure
        if r < N - 1:
            new_h = F.next_hint(x.question, history)
            h = new_h or h

    # fallback: if no hint strictly beat baseline, return the last one tried
    if best_hint is None and h is not None:
        best_hint = h

    return best_hint


# =====================================================================
# Algorithm 2 -- FGPO-RLOO  (outer training loop)
# =====================================================================
def fgpo_train(
    D_train, D_test,    # train / test problem sets
    pi_theta,           # policy (LoRA over a frozen base LM)
    F,                  # frontier model
    R,                  # oracle reward fn
    n_epochs: int,
    B: int,             # problems per batch
    G: int,             # RLOO group size = num_generations per row
    beta: float,        # KL coefficient
    eta: float,         # learning rate
    rho: float,         # fraction of D_train eligible for hints
):
    C = HintCache()                                                # pid -> h* (or None)
    E = {p.id for p in D_train[: int(rho * len(D_train))]}         # hint-eligible set

    for epoch in range(n_epochs):
        for batch in chunk(D_train, B):

            # (1) lazy hint construction: run Algorithm 1 once per eligible problem
            for x in batch:
                if x.id in E and x.id not in C:
                    C[x.id] = step1_loop(x, pi_theta, F, R,
                                         N=3, K=10, tau=0.5,
                                         worst_k=3, best_k=1)

            # (2) build augmented batch
            #     - every problem contributes a no-hint row
            #     - eligible problems with a usable hint contribute a 2nd, hinted row
            #     - the two rows are independent RLOO groups (rollouts are NOT mixed)
            rows = []
            for x in batch:
                rows.append(make_row(x, hint=None))
                h = C.get(x.id)
                if x.id in E and h is not None:
                    rows.append(make_row(x, hint=h))

            # (3) RLOO update: for each row sample G completions y_1..y_G ~ pi_theta,
            #     compute leave-one-out advantage  A_i = R(y_i,x) - mean_{j!=i} R(y_j,x),
            #     and minimize  L = -E_i[A_i * log pi_theta(y_i|row)] + beta * KL(pi_theta||pi_ref).
            pi_theta = rloo_step(pi_theta, rows, G=G, beta=beta, lr=eta, R=R)

            # (4) held-out eval is hint-free -- this is what we report
            log_test_accuracy(pi_theta, D_test)
