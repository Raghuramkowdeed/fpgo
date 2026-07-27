# SDFT trajectories — one simple example per dataset

Every prompt below was produced by the **training code itself** (`build_student_messages`,
`build_teacher_messages`, `MemBank.retrieve` from `run_fused.py`) applied to the run's own
saved artifacts. Demos are retrieved only from problems seen in **earlier** batches.
Long option lists / wiki contexts are summarized as `[...]`; the unclipped records are in
`sdft_traces_sample.jsonl`.

**Flow for every problem:** arrive → retrieve demos from past → answer (student prompt,
no hint) → oracle grades 0/1 (**this is the recorded score, final**) → if correct, the
model's own answer is stored → for the next ~9 batches, training uses teacher prompt =
student prompt + that stored self-answer as hint. Gold labels never leave the oracle.


## DDXPlus — medical diagnosis  (`ddxplus#606`, arrived at stream position 1607)

**1. Retrieval** — memory search over past problems returned: `ddxplus#199`, `ddxplus#440`, `ddxplus#441`

**2. Student prompt** — this is everything the model saw when it was scored:

```text
SYSTEM: You are an expert medical doctor making a differential diagnosis.

USER      (past problem): patient profile: ""Sex: Male, Age: 56
- I have been in contact with someone who has had pertussis (whooping cough).
- I vomited after coughing.
- I have intense coughing fits.
- My vaccinations are up to date.""
ASSISTANT (its own past verified answer): 48. Whooping cough

USER      (past problem): patient profile: ""Sex: Female, Age: 22
- I have been in contact with someone who has had pertussis (whooping cough).
- I vomited after coughing.
- My vaccinations are up to date.""
ASSISTANT (its own past verified answer): 48. Whooping cough

USER      (past problem): patient profile: ""Sex: Male, Age: 7
- I have been in contact with someone who has had pertussis (whooping cough).
- I have asthma or have used a bronchodilator in the past.
- I vomited after coughing.
- I have intense coughing fits.
- My vaccinations are up to date.""
ASSISTANT (its own past verified answer): 48. Whooping cough

USER (the new problem): Act as a medical doctor and diagnose the patient based on the following patient profile:

patient profile: ""Sex: Male, Age: 28
- I vomited after coughing.
- My vaccinations are up to date.""

All possible diagnoses ... [list of 49 options: 0. Acute COPD exacerbation / infection; 1. Acute dystonic reactions; 2. Acute laryngitis; ... ] 
Now, directly provide the diagnosis for the patient in the following format: <number>. <diagnosis>
```

**3. Oracle** — the model's answer was graded **CORRECT** → recorded, final.

**4. Stored** — the model's own answer goes into memory:

```text
48. Whooping cough
```

**5. Teacher prompt (training only)** — identical to the student prompt, plus:

```text
ASSISTANT (hint = the stored self-answer): 48. Whooping cough
USER: (the same problem, repeated)
```

The student never sees the hint; forward KL teaches it to match the teacher's
hint-informed distribution. The hint is the model's own output from step 3-4 —
not a dataset field.


## HotpotQA — multi-hop QA  (`hotpotqa#400`, arrived at stream position 1191)

**1. Retrieval** — memory search over past problems returned: `hotpotqa#107`, `hotpotqa#115`, `hotpotqa#179`

**2. Student prompt** — this is everything the model saw when it was scored:

```text
SYSTEM: You are a careful assistant answering multi-hop questions with short, exact answers in JSON.

USER      (past problem): Are Ferocactus and Silene both types of plant?
ASSISTANT (its own past verified answer): {"answer": "Yes"}

USER      (past problem): Are Pothos and Tibouchina Aubl both flowering genus of plants?
ASSISTANT (its own past verified answer): {"answer": "Yes"}

USER      (past problem): Both Sorbaria and Goldenrod are a genus of more than 50 flowering plants?
ASSISTANT (its own past verified answer): {"answer": "No"}

USER (the new problem): You are doing a question-answering task. You are given the following context, which might help you answer the question:
Context: [2 Wikipedia paragraphs: Stapelianthus, Cistus, ...]
Question: Cistus and Stapelianthus are both forms of what?
Now, provide your answer in the following JSON format:
{"answer": "<your answer text span>"}
```

**3. Oracle** — the model's answer was graded **CORRECT** → recorded, final.

**4. Stored** — the model's own answer goes into memory:

```text
{"answer": "Flowering plants"}
```

**5. Teacher prompt (training only)** — identical to the student prompt, plus:

```text
ASSISTANT (hint = the stored self-answer): {"answer": "Flowering plants"}
USER: (the same problem, repeated)
```

The student never sees the hint; forward KL teaches it to match the teacher's
hint-informed distribution. The hint is the model's own output from step 3-4 —
not a dataset field.


## DS-1000 — data-science code  (`ds1000#537`, arrived at stream position 2891)

**1. Retrieval** — memory search over past problems returned: `ds1000#581`, `ds1000#536`, `ds1000#636`

**2. Student prompt** — this is everything the model saw when it was scored:

```text
SYSTEM: You are an expert Python programmer for data science. Solve the problem by writing correct Python code. Output ONLY the code in a ```python ... ``` block.

USER      (past problem): import numpy as np
import matplotlib.pyplot as plt


H = np.random.randn(10, 10)

# color plot of the 2d array H
# SOLUTION START
ASSISTANT (its own past verified answer): ```python

plt.imshow(H)




```

USER      (past problem): import matplotlib.pyplot as plt
import numpy as np

data = np.random.random((10, 10))

# plot the 2d matrix data with a colorbar
# SOLUTION START
ASSISTANT (its own past verified answer): ```python


plt.imshow(data)
plt.colorbar()






```

USER      (past problem): import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.rand(10)
y = np.random.rand(10)

# Make a histogram of x and show outline of each bar in the histogram
# Make the outline of each bar has a line width of 1.2
# SOLUTION START
ASSISTANT (its own past verified answer): ```python



plt.hist(x, edgecolor='black', linewidth=1.2)



```

USER (the new problem): Here is the user's requirements for solving a programming problem (enclosed in '''):
'''
import numpy as np
import matplotlib.pyplot as plt

H = np.random.randn(10, 10)

# show the 2d array H in black and white
# SOLUTION START
'''

[... followed by boilerplate showing how the code will be tested (exec harness) ...]

Now, generate your code directly in the following format:
```python
<your_code>
```
```

**3. Oracle** — the model's answer was graded **CORRECT** → recorded, final.

**4. Stored** — the model's own answer goes into memory:

```text
```python

plt.imshow(H, cmap='gray')

```
```

**5. Teacher prompt (training only)** — identical to the student prompt, plus:

```text
ASSISTANT (hint = the stored self-answer): ```python

plt.imshow(H, cmap='gray')

```
USER: (the same problem, repeated)
```

The student never sees the hint; forward KL teaches it to match the teacher's
hint-informed distribution. The hint is the model's own output from step 3-4 —
not a dataset field.
