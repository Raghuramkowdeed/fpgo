# Positioning memo A — LLM self-improvement / self-training (verified via web, Aug 2026)

## Works
| Work | Venue/Year | ~Cites | Contribution | Gap vs ours |
|---|---|---|---|---|
| STaR (Zelikman et al.) | NeurIPS 2022 | ~1000 | FT on self-generated rationales that yield correct answers; "rationalization" = hint-conditioned generation | Offline multi-epoch, restarts from base each iter, fixed labeled set; no stream/memory |
| ReST (Gulcehre et al.) | arXiv 2023 | ~480 | Grow/Improve: sample, reward-filter, offline FT | Reward model + repeated offline passes; nothing learned mid-stream |
| ReST-EM (Singh et al.) | TMLR 2024 | ~340 | ReST as EM with binary verifier; scales past human data | Our signal exactly, but few full-corpus EM rounds; needs all problems upfront |
| RFT (Yuan et al.) | arXiv 2023 | ~360 | Rejection-sampling FT; distinct correct paths drive gains | One-shot offline augmentation w/ gold answers for filtering |
| SPIN (Chen et al.) | ICML 2024 | ~610 | Self-play DPO vs own previous generations | Requires human SFT corpus as target; full-dataset iterations |
| Self-Rewarding LMs (Yuan et al.) | ICML 2024 | ~670 | LLM-as-judge builds preference pairs for iterative DPO | Unverified self-judgment; offline iterative |
| Self-Instruct (Wang et al.) | ACL 2023 | ~3480 | Bootstrap instruction corpus from model | Data creation for one SFT run; heuristic filter |
| RAFT (Dong et al.) | TMLR 2023 | ~755 | Best-of-n rank by RM, SFT on top | Offline batches, learned scalar RM |
| SEAL (Zweiger et al., incl. Agrawal) | NeurIPS 2025 | ~50 | Model emits self-edits (own FT data) rewarded via RL | Needs repeated eval episodes; reports forgetting as open — our memory+LoRA targets this |
| TTRL (Zuo et al.) | NeurIPS 2025 | ~210 | RL on test data w/ majority-vote pseudo-labels | Ungrounded consensus reward, many epochs over one test set, collapses when majority wrong |
| Absolute Zero (Zhao et al.) | NeurIPS 2025 | ~280 | Self-play task proposal + executor verifier, zero data | Pre-deployment self-play on invented tasks, not an exogenous stream |

## Strand summary
- Established: filter own samples by correctness signal + FT works (STaR→ReST/ReST-EM/RFT/RAFT→SPIN/Self-Rewarding→AZR/TTRL). Gains ∝ coverage/diversity of correct samples. Long iteration causes drift → patched by base-restart.
- Open: improving WHILE serving a one-pass stream; permanent per-instance weight updates without corpus replay, base-restarts, or forgetting (SEAL names forgetting open; TTRL collapses).
- Sharpest positioning: "Prior self-training improves a model BETWEEN deployments — sampling, filtering, re-training over a static corpus for multiple epochs — whereas we show a 7B model can improve itself DURING deployment: a single pass in which each verified success is immediately written to memory for retrieval and distilled into LoRA weights, so the model answering problem t+1 is permanently better than the one that answered problem t, with only binary feedback, no gold labels, no teacher."

## STaR intro template (para-by-para rhetorical moves)
1 cognitive hook → 2 phenomenon established w/ prior evidence → 3 dead-end A (hand-built rationale datasets expensive) → 4 dead-end B (few-shot prompting underperforms) → 5 the turn ("we adopt a different approach": the simple loop) → 6 the loop's own failure mode (can't learn from unsolved problems) motivates key component (rationalization/hint) → 7 named method + figure → 8 headline numbers → contributions list.
Our mapping: dead-end A = offline multi-epoch self-training can't run on a stream; dead-end B = memory-only ICL never improves the model / scalar-reward RL is thin+unstable; para-6 failure = naive online training on greedy successes stalls (coverage) → motivates reward-filtered memory + hint self-distillation + forward re-eval.

## Verified BibTeX (reconcile keys with refs.bib during rewrite)
@inproceedings{zelikman2022star, title={{STaR}: Bootstrapping Reasoning With Reasoning}, author={Zelikman, Eric and Wu, Yuhuai and Mu, Jesse and Goodman, Noah D.}, booktitle={NeurIPS}, year={2022}}
@article{gulcehre2023rest, title={Reinforced Self-Training ({ReST}) for Language Modeling}, author={Gulcehre, Caglar and Paine, Tom Le and Srinivasan, Srivatsan and Konyushkova, Ksenia and Weerts, Lotte and Sharma, Abhishek and Siddhant, Aditya and Ahern, Alex and Wang, Miaosen and Gu, Chenjie and Macherey, Wolfgang and Doucet, Arnaud and Firat, Orhan and de Freitas, Nando}, journal={arXiv preprint arXiv:2308.08998}, year={2023}}
@article{singh2024beyond, title={Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models}, author={Singh, Avi and Co-Reyes, John D. and Agarwal, Rishabh and Anand, Ankesh and Patil, Piyush and Garcia, Xavier and Liu, Peter J. and Harrison, James and Lee, Jaehoon and Xu, Kelvin and Parisi, Aaron and others}, journal={TMLR}, year={2024}}
@article{yuan2023scaling, title={Scaling Relationship on Learning Mathematical Reasoning with Large Language Models}, author={Yuan, Zheng and Yuan, Hongyi and Li, Chengpeng and Dong, Guanting and Lu, Keming and Tan, Chuanqi and Zhou, Chang and Zhou, Jingren}, journal={arXiv preprint arXiv:2308.01825}, year={2023}}
@inproceedings{chen2024spin, title={Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models}, author={Chen, Zixiang and Deng, Yihe and Yuan, Huizhuo and Ji, Kaixuan and Gu, Quanquan}, booktitle={ICML}, year={2024}}
@inproceedings{yuan2024selfrewarding, title={Self-Rewarding Language Models}, author={Yuan, Weizhe and Pang, Richard Yuanzhe and Cho, Kyunghyun and Li, Xian and Sukhbaatar, Sainbayar and Xu, Jing and Weston, Jason}, booktitle={ICML}, year={2024}}
@inproceedings{wang2023selfinstruct, title={Self-Instruct: Aligning Language Models with Self-Generated Instructions}, author={Wang, Yizhong and Kordi, Yeganeh and Mishra, Swaroop and Liu, Alisa and Smith, Noah A. and Khashabi, Daniel and Hajishirzi, Hannaneh}, booktitle={ACL}, year={2023}}
@article{dong2023raft, title={{RAFT}: Reward rAnked FineTuning for Generative Foundation Model Alignment}, author={Dong, Hanze and Xiong, Wei and Goyal, Deepanshu and Zhang, Yihan and Chow, Winnie and Pan, Rui and Diao, Shizhe and Zhang, Jipeng and Shum, KaShun and Zhang, Tong}, journal={TMLR}, year={2023}}
@inproceedings{zweiger2025seal, title={Self-Adapting Language Models}, author={Zweiger, Adam and Pari, Jyothish and Guo, Han and Aky{\"u}rek, Ekin and Kim, Yoon and Agrawal, Pulkit}, booktitle={NeurIPS}, year={2025}}
@inproceedings{zuo2025ttrl, title={{TTRL}: Test-Time Reinforcement Learning}, author={Zuo, Yuxin and Zhang, Kaiyan and Sheng, Li and Qu, Shang and Cui, Ganqu and Zhu, Xuekai and Li, Haozhan and Zhang, Yuchen and Long, Xinwei and Hua, Ermo and Qi, Biqing and Wang, Youbang and Ding, Ning and Zhou, Bowen}, booktitle={NeurIPS}, year={2025}}
@inproceedings{zhao2025absolute, title={Absolute Zero: Reinforced Self-play Reasoning with Zero Data}, author={Zhao, Andrew and Wu, Yiran and Yue, Yang and Wu, Tong and Xu, Quentin and Lin, Matthieu and Wang, Shenzhi and Wu, Qingyun and Zheng, Zilong and Huang, Gao}, booktitle={NeurIPS}, year={2025}}

Caveats: TTRL author list from arXiv v3; RFT is arXiv-only (cite Llama-2 separately for rejection sampling in production); cite counts are Semantic Scholar (conservative).
