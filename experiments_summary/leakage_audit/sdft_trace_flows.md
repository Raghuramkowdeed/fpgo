# SDFT trajectory of a new question — real examples, verbatim

Each example below is a problem whose memory entry was **never overwritten**, so the
stored text is byte-identical to what the model generated when the question arrived.
Prompts are rebuilt by the training code itself (`build_student_messages`,
`MemBank.retrieve` from `run_fused.py`), demos restricted to earlier batches.


---

## DDXPlus (medical diagnosis) — `ddxplus#1386`

**1. New question arrives** (stream position 3346, batch 334):

```text
Act as a medical doctor and diagnose the patient based on the following patient profile:

patient profile: ""Sex: Female, Age: 19
- I take corticosteroids.
- I am currently using intravenous drugs.
- I have diabetes.
- I have a fever (either felt or measured with a thermometer).
- I have had an involuntary weight loss over the last 3 months.
- I have a cough.
- My BMI is less than 18.5, or I am underweight.""
[+ list of 49 diagnosis options]
Now, directly provide the diagnosis for the patient in the following format: <number>. <diagnosis>
```

**2. ICL retrieval** — 3 most similar PAST problems from memory (ddxplus#315`, `ddxplus#511`, `ddxplus#1199), each shown with the model's own earlier verified answer:

```text
Q: patient profile: ""Sex: Female, Age: 25
- I take corticosteroids.
- I am currently using intravenous drugs.
- I have diabetes.
- I have a fever (either felt or measured with a thermometer).
- I have had an involuntary weight loss over the last 3 mont [...]
A (model's past answer): 44. Tuberculosis

Q: patient profile: ""Sex: Female, Age: 21
- I take corticosteroids.
- I am currently using intravenous drugs.
- I am experiencing shortness of breath or difficulty breathing in a significant way.
- I have diabetes.
- I have had an involuntary weight lo [...]
A (model's past answer): 44. Tuberculosis

Q: patient profile: ""Sex: Female, Age: 52
- I take corticosteroids.
- I am currently using intravenous drugs.
- I am experiencing shortness of breath or difficulty breathing in a significant way.
- I have diabetes.
- I drink alcohol excessively or I ha [...]
A (model's past answer): 44. Tuberculosis

```

**3. Model generates its answer** (greedy, from system + demos + question — nothing else):

```text
44. Tuberculosis
```

**4. Oracle evaluation** — string-match: predicted diagnosis == dataset's diagnosis label -> 1, else 0
Result: **CORRECT (1)** -> written to the benchmark record. Final.

**5. Stored in mem_bank** (reward filter passed): the answer above becomes this problem's entry —
available as an ICL demo for future questions, and as the training hint:
teacher prompt = the exact prompt from steps 1-2 + this answer appended as an assistant turn.
The student trains to match the teacher's distribution WITHOUT seeing it.


---

## HotpotQA (multi-hop QA) — `hotpotqa#1193`

**1. New question arrives** (stream position 3344, batch 334):

```text
You are doing a question-answering task. You are given the following context, which might help you answer the question:
Context: [10 Wikipedia paragraphs: Martin Linnes, Leigh Broxham, Ibrahim Tall...]
Question: In what year was the team that Luke Clark plays either right back or central midfielder for founded?
```

**2. ICL retrieval** — 3 most similar PAST problems from memory (hotpotqa#109`, `hotpotqa#679`, `hotpotqa#1051), each shown with the model's own earlier verified answer:

```text
Q: The 2011–12 Houston Rockets season began with the drafting of the player who now plays for what team?
A (model's past answer): {"answer": "Boston Celtics"}

Q: Jack Alfred Evans played with a football club team that is based in what city in Australia?
A (model's past answer): {"answer": "Melbourne"}

Q: What was the other Los Angeles team that the 1999 Clippers top draft pick played for and won two championships with?
A (model's past answer): {"answer": "Los Angeles Lakers"}

```

**3. Model generates its answer** (greedy, from system + demos + question — nothing else):

```text
{"answer": "1963"}
```

**4. Oracle evaluation** — exact-match of the answer span vs the dataset's gold span -> 1, else 0
Result: **CORRECT (1)** -> written to the benchmark record. Final.

**5. Stored in mem_bank** (reward filter passed): the answer above becomes this problem's entry —
available as an ICL demo for future questions, and as the training hint:
teacher prompt = the exact prompt from steps 1-2 + this answer appended as an assistant turn.
The student trains to match the teacher's distribution WITHOUT seeing it.


---

## DS-1000 (data-science code) — `ds1000#657`

**1. New question arrives** (stream position 801, batch 80):

```text
Here is the user's requirements for solving a programming problem (enclosed in '''):
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.arange(10)

# plot y over x with label "y"
# make the legend fontsize 8
# SOLUTION START
'''
[+ boilerplate showing the exec-based test harness]
Now, generate your code directly...
```

**2. ICL retrieval** — 3 most similar PAST problems from memory (ds1000#664`, `ds1000#563`, `ds1000#579), each shown with the model's own earlier verified answer:

```text
Q: import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.arange(10)

# Plot y over x in a line chart and label the line "y over x"
# Show legend of the plot and give the legend box a title
# SOLUTION START
A (model's past answer): ```python



plt.plot(x, y, label='y over x')
plt.legend(title='Legend Box Title')









```

Q: import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.arange(10)

# Plot y over x in a line chart and name axis with labels ("x" and "y")
# Hide tick labels but keep axis labels
# SOLUTION START
A (model's past answer): ```python



plt.xlabel('x')
plt.ylabel('y')

plt.xticks([])
plt.yticks([])






```

Q: import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.arange(10)

# Plot y over x
# move the y axis ticks to the right
# SOLUTION START
A (model's past answer): ```python



plt.gca().yaxis.tick_right()








```

```

**3. Model generates its answer** (greedy, from system + demos + question — nothing else):

```text
```python

plt.plot(x, y, label='y')
plt.legend(fontsize=8)

```
```

**4. Oracle evaluation** — EXECUTION: the code is run inside the problem's test harness; all assertions pass -> 1, else 0
Result: **CORRECT (1)** -> written to the benchmark record. Final.

**5. Stored in mem_bank** (reward filter passed): the answer above becomes this problem's entry —
available as an ICL demo for future questions, and as the training hint:
teacher prompt = the exact prompt from steps 1-2 + this answer appended as an assistant turn.
The student trains to match the teacher's distribution WITHOUT seeing it.


---

## Counter-example: wrong answer -> nothing stored

`hotpotqa#12` arrived at stream position 44, the model's answer was graded **WRONG (0)**.
It was never stored in memory: it can never appear as a demo and never becomes a training
hint (unless a later window re-eval solves it). Wrong answers are simply discarded —
1297 of the 3360 problems so far are in this bucket.
