# Positioning memo B — online adaptation / agent memory / test-time learning (verified Aug 2026)

## Works (condensed)
| Work | Venue/Yr | ~Cites | Gap vs ours |
|---|---|---|---|
| Reflexion (Shinn) | NeurIPS 2023 | ~4600 | verbal reflection buffer, explicitly "not by updating weights"; resets with buffer |
| Voyager (Wang G.) | TMLR 2024 | ~2050 | skill library around frozen GPT-4; "bypasses fine-tuning" |
| MemGPT (Packer) | arXiv 2023 | ~1020 | memory infra only, no feedback, nothing learned |
| MemPrompt (Madaan) | EMNLP 2022 | ~200-300 (GS) | retrieval-from-feedback ICL ancestor; never distilled to weights |
| StreamBench (Wu) | NeurIPS 2024 D&B | ~40 | our testbed; its conclusion = memory methods win → we overturn |
| Evo-Memory (Wei) | arXiv 2511.20857 | new | streaming self-evolving MEMORY (not params) — confirms community framing |
| TTT (Sun) | ICML 2020 | 1000+ | weight updates but transient, self-supervised, robustness-aimed |
| TTT-ARC (Akyürek) | ICML 2025 | ~80 | per-instance LoRA then DISCARDED; no stream/verifier/accumulation |
| SIFT (Hübotter) | ICLR 2025 | ~45 | active test-time FT on retrieved corpus data; transient, rewardless |
| CL survey (Shi) | ACM CSUR 2025 | ~320 | CL assumes curated phase-wise datasets, not self-generated online supervision |
| O-LoRA (Wang X.) | Findings EMNLP 2023 | ~335 | needs task boundaries + labels |
| Forgetting (Luo) | arXiv 2308.08747 | ~745 | the standard objection to naive online FT; our low-KL self-distill sidesteps |
| ExpeL (Zhao) | AAAI 2024 | ~730 | NL insights, "without parameter updates" |
| AWM (Wang Z.) | ICML 2025 | ~210 | nearest neighbor: stream + self-verified successes → but stored as TEXT workflows |
| ACE (Zhang Q.) | arXiv 2510.04618 | new | "context engineering INSTEAD of parameter updates" — the program our results argue against |

## Strand summary
- Established: post-deployment improvement happens via EXTERNAL stores (buffers/libraries/playbooks); streaming eval canonized by StreamBench (memory baselines strongest). Test-time weight adaptation works but is per-instance + discarded. Naive continual FT forgets (Luo); CL methods need task-segmented labels.
- Open: permanent cumulative weight improvement from binary verifiable feedback one problem at a time; memory-as-hint-source for distillation essentially unstudied.
- Sharpest sentence: "Prior streaming self-improvement stores what the model learned OUTSIDE the model — buffers, libraries, playbooks — while test-time training updates weights only to discard them; we distill each verified success from retrieval memory into the weights themselves, yielding a permanently improving model that outperforms retrieval-ICL on StreamBench and remains stable under distribution shift where REINFORCE++ collapses."

## Reflexion intro template (4 paras)
1 incumbent economically infeasible (RL needs samples+finetuning → agents stuck with ICL) + one-sentence pivot to proposal → 2 mechanism + human analogy → 3 concede difficulty, taxonomize the design space, bullet advantages → 4 concrete deltas + artifact + contribution list.
Our mapping: 1 = "streaming agents improve via memory because weight updates deemed costly/unstable"; 3 = concede forgetting/RL-instability, taxonomy {context / transient-weights / permanent-weights (ours)}, advantages bullets; 4 = numbers + contributions.

## BibTeX (verified; full entries)
@inproceedings{shinn2023reflexion, title={Reflexion: Language Agents with Verbal Reinforcement Learning}, author={Shinn, Noah and Cassano, Federico and Berman, Edward and Gopinath, Ashwin and Narasimhan, Karthik and Yao, Shunyu}, booktitle={NeurIPS}, year={2023}}
@article{wang2024voyager, title={Voyager: An Open-Ended Embodied Agent with Large Language Models}, author={Wang, Guanzhi and Xie, Yuqi and Jiang, Yunfan and Mandlekar, Ajay and Xiao, Chaowei and Zhu, Yuke and Fan, Linxi and Anandkumar, Anima}, journal={TMLR}, year={2024}}
@article{packer2023memgpt, title={MemGPT: Towards LLMs as Operating Systems}, author={Packer, Charles and Wooders, Sarah and Lin, Kevin and Fang, Vivian and Patil, Shishir G. and Stoica, Ion and Gonzalez, Joseph E.}, journal={arXiv preprint arXiv:2310.08560}, year={2023}}
@inproceedings{madaan2022memory, title={Memory-assisted prompt editing to improve {GPT}-3 after deployment}, author={Madaan, Aman and Tandon, Niket and Clark, Peter and Yang, Yiming}, booktitle={EMNLP}, pages={2833--2861}, year={2022}}
@inproceedings{sun2020test, title={Test-Time Training with Self-Supervision for Generalization under Distribution Shifts}, author={Sun, Yu and Wang, Xiaolong and Liu, Zhuang and Miller, John and Efros, Alexei and Hardt, Moritz}, booktitle={ICML}, year={2020}}
@inproceedings{hubotter2025efficiently, title={Efficiently Learning at Test-Time: Active Fine-Tuning of {LLMs}}, author={H{\"u}botter, Jonas and Bongni, Sascha and Hakimi, Ido and Krause, Andreas}, booktitle={ICLR}, year={2025}}
@inproceedings{akyurek2025surprising, title={The Surprising Effectiveness of Test-Time Training for Few-Shot Learning}, author={Aky{\"u}rek, Ekin and Damani, Mehul and Zweiger, Adam and Qiu, Linlu and Guo, Han and Pari, Jyothish and Kim, Yoon and Andreas, Jacob}, booktitle={ICML}, year={2025}}
@article{shi2025continual, title={Continual Learning of Large Language Models: A Comprehensive Survey}, author={Shi, Haizhou and Xu, Zihao and Wang, Hengyi and Qin, Weiyi and Wang, Wenyuan and Wang, Yibin and Wang, Zifeng and Ebrahimi, Sayna and Wang, Hao}, journal={ACM Computing Surveys}, year={2025}}
@inproceedings{wang2023orthogonal, title={Orthogonal Subspace Learning for Language Model Continual Learning}, author={Wang, Xiao and Chen, Tianze and Ge, Qiming and Xia, Han and Bao, Rong and Zheng, Rui and Zhang, Qi and Gui, Tao and Huang, Xuanjing}, booktitle={Findings of EMNLP}, year={2023}}
@article{luo2023empirical, title={An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning}, author={Luo, Yun and Yang, Zhen and Meng, Fandong and Li, Yafu and Zhou, Jie and Zhang, Yue}, journal={arXiv preprint arXiv:2308.08747}, year={2023}}
@inproceedings{zhao2024expel, title={ExpeL: {LLM} Agents Are Experiential Learners}, author={Zhao, Andrew and Huang, Daniel and Xu, Quentin and Lin, Matthieu and Liu, Yong-Jin and Huang, Gao}, booktitle={AAAI}, year={2024}}
@inproceedings{wang2025agent, title={Agent Workflow Memory}, author={Wang, Zora Zhiruo and Mao, Jiayuan and Fried, Daniel and Neubig, Graham}, booktitle={ICML}, year={2025}}
@article{wei2025evomemory, title={Evo-Memory: Benchmarking {LLM} Agent Test-time Learning with Self-Evolving Memory}, author={Wei, Tianxin and others}, journal={arXiv preprint arXiv:2511.20857}, year={2025}}
@article{zhang2025agentic, title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models}, author={Zhang, Qizheng and others}, journal={arXiv preprint arXiv:2510.04618}, year={2025}}

Caveats: MemPrompt S2 count fragmented (use "200-300, GS"); TTT "1,000+"; Akyürek camera-ready title changed from arXiv v1 — cite ICML title. Reflexion author list here (6 authors incl. Berman) supersedes our current refs.bib entry (5 authors) — fix during bib merge.
