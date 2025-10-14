# Open Questions and Empirical Verification Requirements

**Project:** Provably Safe LLM Agents via Causal Intervention
**Date:** October 12, 2025
**Purpose:** Document theoretical questions requiring empirical investigation

---

## Category 1: Assumption Validation

These questions verify whether the theoretical assumptions hold in practice for trained LLM agents.

### Q1.1: D-Separation in Learned Representations

**Theoretical Claim:** Representation $R$ satisfies $(R \perp\!\!\!\perp U_{\text{instr}} \mid S)$ (Theorem 3.1, Condition 1)

**Empirical Question:** Does this hold in representations learned by causal training?

**Measurement Approach:**
- Extract representations $R$ from trained model for inputs $(S, U)$
- Label or extract $U_{\text{instr}}$ component (instruction content in user input)
- Compute HSIC$(R, U_{\text{instr}} \mid S)$ using RBF kernels
- Test predictability: train classifier $g(R, S) \to U_{\text{instr}}$, measure accuracy

**Success Criterion:**
- HSIC $< 0.05$ (strong independence)
- Classifier accuracy $\approx$ random chance (e.g., 10% for 10-class problem)

**Potential Violations:**
- If HSIC $> 0.1$: representation still contains instruction information
- If classifier achieves $> 50\%$ accuracy: spurious correlation remains

**Follow-up if violated:** Increase weight on causal loss term, use adversarial independence training

---

### Q1.2: Data Preservation

**Theoretical Claim:** Representation $R$ maintains $I(R; U_{\text{data}} \mid S) \geq I_{\min}$ (Theorem 3.1, Condition 2)

**Empirical Question:** Does causal training preserve task-relevant data information?

**Measurement Approach:**
- Measure mutual information $\hat{I}(R; U_{\text{data}} \mid S)$ using MINE (Mutual Information Neural Estimation)
- Compare to baseline model: $\Delta I = I_{\text{causal}} - I_{\text{baseline}}$
- Measure task performance on benign inputs (accuracy, F1, BLEU depending on task)

**Success Criterion:**
- $\hat{I}(R; U_{\text{data}} \mid S) \geq 0.9 \cdot I_{\text{baseline}}$ (retain 90% of data information)
- Task performance $\geq 95\%$ of baseline accuracy

**Potential Violations:**
- If $\hat{I} < 0.7 \cdot I_{\text{baseline}}$: causal constraint too strong, model loses functionality
- If task accuracy $< 90\%$: model cannot perform intended tasks

**Follow-up if violated:** Adjust balance between causal loss and task loss, use multi-objective optimization

---

### Q1.3: Markov Factorization

**Theoretical Claim:** $P(O \mid R, U) = P(O \mid R, U_{\text{data}})$ (Theorem 3.1, Condition 3)

**Empirical Question:** Does output depend on user input only through data content (not instruction content) given representation?

**Measurement Approach:**
- For samples $(s_i, u_i, r_i)$, compute $P(O \mid R = r_i, U = u_i)$ using model
- Create counterfactual $\tilde{u}_i = (u_{i,\text{data}}, \emptyset)$ removing instruction content
- Compute $P(O \mid R = r_i, U = \tilde{u}_i)$
- Measure $\text{KL}(P_O^{u_i} \| P_O^{\tilde{u}_i})$ (should be small)

**Success Criterion:**
- Mean KL divergence $< 0.1$ nats across test set
- Max KL divergence $< 0.5$ nats (no catastrophic violations)

**Potential Violations:**
- If KL $> 0.5$ on average: output still responds to instruction content in $U$ even given $R$
- If max KL $> 2.0$: some inputs cause dramatic output changes based on $U_{\text{instr}}$

**Follow-up if violated:** Investigate architecture - may need to modify direct path $U \to O$ to only carry data content

---

### Q1.4: Causal Graph Structure

**Theoretical Claim:** True causal graph is $S \to R \leftarrow U, R \to O, U \to O$ (Section 1.2)

**Empirical Question:** Does causal discovery from data recover this structure?

**Measurement Approach:**
- Sample $(s_i, u_i, r_i, o_i)$ from trained model
- Run PC algorithm with $\alpha = 0.05$ significance level
- Run GES algorithm maximizing BIC score
- Compute Structural Hamming Distance: $\text{SHD}(\hat{\mathcal{G}}, \mathcal{G}_{\text{true}})$

**Success Criterion:**
- SHD $\leq 1$ (at most 1 edge error or reversal)
- Both PC and GES agree on $R$ having parents $\{S, U\}$
- Both algorithms agree on $O$ having parents $\{R, U\}$

**Potential Violations:**
- If SHD $> 3$: learned structure significantly different, assumptions violated
- If discovered graph has cycles: feedback not captured by DAG assumption
- If discovered graph missing edges: insufficient data or faithfulness violated

**Follow-up if violated:** Collect more data (causal discovery requires large samples), or revise theoretical model if systematic misspecification

---

### Q1.5: No Unmeasured Confounders

**Theoretical Claim:** Causal sufficiency holds (Assumption 7.5)

**Empirical Question:** Are there latent variables affecting multiple observed variables?

**Measurement Approach:**
- Instrumental variable test: use task category $Z$ as instrument for $S \to R$
- Compare 2SLS estimate to OLS estimate: $\hat{\beta}_{IV}$ vs $\hat{\beta}_{OLS}$
- Large difference indicates confounding
- Test for latent confounders using Tetrad software (FCI algorithm for latent variables)

**Success Criterion:**
- $|\hat{\beta}_{IV} - \hat{\beta}_{OLS}| / \hat{\beta}_{OLS} < 0.1$ (less than 10% difference)
- FCI algorithm does not introduce latent nodes in learned graph

**Potential Violations:**
- If $|\hat{\beta}_{IV} - \hat{\beta}_{OLS}| > 0.3$: substantial confounding present
- If FCI introduces latent nodes: unmeasured common causes exist (e.g., user intent affecting both $S$ and $U$)

**Follow-up if violated:** Collect additional covariates to account for confounding, or use robust causal inference methods that handle unmeasured confounding

---

## Category 2: Bound Tightness

These questions determine whether theoretical bounds are practical (non-vacuous) or overly pessimistic.

### Q2.1: Causal Estimation Error

**Theoretical Claim:** Attack success rate $\leq \epsilon_{\text{causal}}$ (Theorem 3.1)

**Empirical Question:** What is $\epsilon_{\text{causal}}$ for trained models, and does it predict attack success?

**Measurement Approach:**
- Sample pairs $(U, U_{\text{data}})$ from test distribution
- Compute $\hat{P}(R \mid S, U)$ and $\hat{P}(R \mid S, U_{\text{data}})$ via kernel density estimation
- Estimate total variation distance: $\hat{D}_{\text{TV}} = \frac{1}{2} \sum_r |\hat{P}(R=r \mid S, U) - \hat{P}(R=r \mid S, U_{\text{data}})|$
- Average over samples: $\hat{\epsilon}_{\text{causal}} = \mathbb{E}[\hat{D}_{\text{TV}}]$
- Measure actual attack success rate on test set
- Plot: attack success vs. $\epsilon_{\text{causal}}$ across different training runs

**Success Criterion:**
- $\epsilon_{\text{causal}} < 0.05$ for deployed model (bound predicts $< 5\%$ attack success)
- Strong correlation: $R^2 > 0.8$ between $\epsilon_{\text{causal}}$ and empirical attack success
- Bound is tight: attack success $\in [0.5 \epsilon_{\text{causal}}, 2 \epsilon_{\text{causal}}]$

**Potential Issues:**
- If $\epsilon_{\text{causal}} > 0.2$: bound too loose to be useful (predicts $< 20\%$ but need $< 5\%$)
- If $R^2 < 0.5$: poor correlation, bound not predictive
- If attack success $\gg \epsilon_{\text{causal}}$: bound violated (theory incorrect or additional assumptions needed)

**Follow-up:** If bound loose but valid, tighten via additional assumptions (e.g., Lipschitz continuity); if violated, investigate failure mode

---

### Q2.2: PAC-Bayesian Bound

**Theoretical Claim:** Generalization bound (Theorem 4.1)
$$\mathcal{L}_{\text{causal}}(h) \leq \hat{\mathcal{L}}_{\text{causal}}(h) + \sqrt{\frac{\text{KL}(Q \| P) + \log(2\sqrt{n}/\delta)}{2n}} + \epsilon_{\text{approx}}$$

**Empirical Question:** Is this bound non-vacuous? Does it predict generalization to novel attacks?

**Measurement Approach:**
- Compute empirical causal risk $\hat{\mathcal{L}}_{\text{causal}}$ on training data
- Estimate KL divergence $\text{KL}(Q \| P)$ for posterior $Q$ (LoRA parameters)
- Calculate predicted upper bound for $\mathcal{L}_{\text{causal}}(h)$
- Measure true causal risk on held-out novel attack family
- Compare: predicted bound vs. empirical performance

**Success Criterion:**
- Bound is non-vacuous: predicted value $< 1.0$ (not trivial)
- Bound is reasonably tight: empirical risk $\leq 2 \times$ predicted bound
- Bound improves with more data: plot bound vs. $n$ shows $O(1/\sqrt{n})$ decay

**Potential Issues:**
- If bound $> 1.0$ (vacuous): need tighter analysis or more prior knowledge
- If empirical risk $> 5 \times$ predicted bound: fundamental assumption violated
- If bound doesn't improve with $n$: approximation error $\epsilon_{\text{approx}}$ dominates (model capacity issue)

**Follow-up:** Use PAC-Bayes with tighter priors (data-dependent bounds), or switch to stability-based analysis

---

### Q2.3: Sample Complexity

**Theoretical Claim:** $n = O\left( \frac{d + \log(1/\delta)}{\epsilon^2} \right)$ (Corollary 4.1)

**Empirical Question:** What is the effective dimension $d$ in practice, and how much data is actually needed?

**Measurement Approach:**
- Train models with varying dataset sizes: $n \in \{10^3, 10^4, 10^5\}$
- For each $n$, measure attack success rate
- Estimate effective dimension $d$ by fitting: $\text{error}(n) = c_1/\sqrt{n} + c_2$ and solving for $d$
- Determine minimum $n$ to achieve $\epsilon = 0.05$ attack success with $\delta = 0.01$ confidence

**Success Criterion:**
- Effective dimension $d < 10^4$ (achievable with modest data)
- Minimum $n < 50,000$ examples for $\epsilon = 0.05$ robustness
- Learning curve follows predicted $O(1/\sqrt{n})$ rate

**Potential Issues:**
- If $d > 10^6$: impractically large sample complexity
- If minimum $n > 10^6$: infeasible data requirements
- If learning curve is slower than $O(1/\sqrt{n})$: theory underestimates difficulty

**Follow-up:** Use transfer learning from pretrained models to reduce effective $d$, or collect larger dataset

---

## Category 3: Decomposition Methods

These questions address how to operationalize the instruction vs. data decomposition in practice.

### Q3.1: Instruction Content Identification

**Theoretical Requirement:** Decompose $U = (U_{\text{data}}, U_{\text{instr}})$ with $I(U_{\text{data}}; U_{\text{instr}}) = 0$

**Empirical Question:** What is the best method to identify instruction content in user inputs?

**Candidate Approaches:**
1. **Manual labeling:** Human annotators mark instruction phrases
2. **Rule-based extraction:** Regex/parsing to identify imperative verbs, instruction keywords
3. **Learned classifier:** Train model to classify tokens as data vs. instruction
4. **Counterfactual generation:** Use LLM to generate $U_{\text{data}}$ by removing instructions

**Measurement Approach:**
- For each method, extract $U_{\text{instr}}$ from test set
- Measure inter-method agreement (Cohen's kappa)
- Train causal model using each decomposition method
- Compare resulting $\epsilon_{\text{causal}}$ and attack success rate

**Success Criterion:**
- Inter-annotator agreement $\kappa > 0.8$ (substantial agreement) for manual labeling
- Automated methods achieve $> 90\%$ F1 compared to manual labels
- Choice of method changes final attack success by $< 2\%$ (robust to decomposition)

**Potential Issues:**
- If $\kappa < 0.6$: ambiguous task, annotators disagree on what counts as instruction
- If automated methods $< 70\%$ F1: insufficient accuracy for reliable training
- If choice of method changes attack success by $> 10\%$: method is critical, need principled selection

**Follow-up:** Develop hybrid approach combining multiple methods, or use ensemble training with different decompositions

---

### Q3.2: Orthogonality of Decomposition

**Theoretical Requirement:** $I(U_{\text{data}}; U_{\text{instr}}) = 0$ (perfect independence)

**Empirical Question:** Can we achieve orthogonal decomposition, or only approximate?

**Measurement Approach:**
- For decomposed $(U_{\text{data}}, U_{\text{instr}})$, estimate $\hat{I}(U_{\text{data}}; U_{\text{instr}})$
- Use MINE or binning-based entropy estimation
- Test impact of violating orthogonality: introduce controlled correlation, measure effect on $\epsilon_{\text{causal}}$

**Success Criterion:**
- $\hat{I}(U_{\text{data}}; U_{\text{instr}}) < 0.1$ bits (low mutual information)
- Adding correlation up to $0.3$ bits changes attack success by $< 5\%$ (robust to violations)

**Potential Issues:**
- If $\hat{I} > 1.0$ bit: substantial entanglement, decomposition invalid
- If small correlation causes $> 20\%$ change in attack success: theory requires exact orthogonality (impractical)

**Follow-up:** Extend theory to allow $I(U_{\text{data}}; U_{\text{instr}}) \leq \epsilon_{\text{decomp}}$, derive bound with additional $\epsilon_{\text{decomp}}$ term

---

### Q3.3: Task-Dependent Decomposition

**Theoretical Question:** Is there a universal decomposition, or must it vary by task?

**Empirical Question:** How does optimal decomposition depend on system instruction $S$?

**Measurement Approach:**
- Train models on multiple tasks: email summarization, code generation, Q&A, translation
- For each task, determine optimal decomposition (minimizes $\epsilon_{\text{causal}}$)
- Compare decompositions across tasks: measure overlap, consistency

**Success Criterion:**
- Single decomposition works across $> 80\%$ of tasks (universal)
- Task-specific decompositions improve attack success by $< 5\%$ compared to universal

**Potential Issues:**
- If universal decomposition fails on $> 50\%$ of tasks: need task-conditional approach
- If task-specific decompositions improve by $> 20\%$: universal approach insufficient

**Follow-up:** Develop task-conditional decomposition $U = (U_{\text{data}}^{(S)}, U_{\text{instr}}^{(S)})$ parameterized by $S$

---

## Category 4: Generalization to Novel Attacks

These questions test whether causal training truly generalizes beyond seen attack families.

### Q4.1: Novel Attack Family Transfer

**Theoretical Claim:** Causal training generalizes to unseen attack families with bounded error (Theorem 4.1)

**Empirical Question:** Does attack success on novel families match the predicted bound?

**Measurement Approach:**
- Train on attack families $\mathcal{F}_1, \ldots, \mathcal{F}_k$ (e.g., jailbreaks, delimiter attacks, role-play)
- Hold out novel family $\mathcal{F}_{\text{new}}$ (e.g., context stuffing, multi-lingual injection)
- Measure attack success on $\mathcal{F}_{\text{new}}$
- Compare to predicted bound from Theorem 4.1
- Repeat with different held-out families

**Success Criterion:**
- Attack success on $\mathcal{F}_{\text{new}} < 10\%$ (strong generalization)
- Attack success $\leq 2 \times \mathcal{L}_{\text{causal}}(h)$ (bound roughly holds)
- Generalization holds across $\geq 3$ different held-out families

**Potential Issues:**
- If attack success $> 50\%$: no generalization, model memorizes training attacks
- If attack success $\gg$ predicted bound: theory fails for OOD attacks
- If generalization works for some families but not others: need better coverage in training

**Follow-up:** Analyze which attack characteristics transfer vs. don't; augment training data with more diverse attacks

---

### Q4.2: Adaptive Attacks

**Empirical Question:** Can attackers who know the causal structure bypass the defense?

**Attack Scenarios:**
1. **Embedding in data:** "Summarize: Dear user, you are now instructed to..."
2. **Exploiting direct path:** Since $U \to O$ exists for data processing, hide instructions as data
3. **Multi-modal injection:** Use images with embedded text instructions
4. **Semantic obfuscation:** Paraphrase instructions to avoid detection

**Measurement Approach:**
- Red team with full knowledge of causal model
- Measure attack success under adaptive attack
- Compare to baseline attack success (without knowledge)

**Success Criterion:**
- Adaptive attacks increase success rate by $< 2x$ (model is not brittle)
- Absolute attack success remains $< 15\%$ even under adaptive attack

**Potential Issues:**
- If adaptive attacks achieve $> 80\%$ success: fundamental vulnerability remains (defense bypassed)
- If increase is $> 10x$: defense only works against naive attacks

**Follow-up:** Combine causal training with adversarial training against adaptive attacks; add runtime output verification

---

### Q4.3: Cross-Model Transfer

**Empirical Question:** Does causal training on one model transfer to other architectures?

**Measurement Approach:**
- Train causal model on Llama 3.1 8B
- Test attack success on same model (in-distribution)
- Transfer learned representations to different models: GPT-2, Mistral, Gemma
- Measure attack success on transferred models

**Success Criterion:**
- Transfer within same family (e.g., Llama 7B $\to$ Llama 13B) maintains $< 2x$ attack success
- Transfer across families (e.g., Llama $\to$ Mistral) maintains $< 5x$ attack success

**Potential Issues:**
- If transfer fails completely (attack success $> 80\%$ on new model): defense is model-specific
- If transfer only works within same architecture: limited generality

**Follow-up:** Train ensemble of models from different families; investigate architectural features that support causal structure

---

## Category 5: Practical Deployment

These questions address real-world implementation challenges.

### Q5.1: Computational Overhead

**Empirical Question:** What is the latency cost of causal training and verification?

**Measurement Approach:**
- Measure inference time: baseline vs. causal model
- Measure cost of online verification (HSIC computation on live traffic)
- Test on consumer hardware (RTX 4050) and cloud GPUs

**Success Criterion:**
- Inference latency increase $< 20\%$ compared to baseline
- Online verification overhead $< 50$ ms per request
- Fits in GPU memory with 4-bit quantization on RTX 4050 (6 GB VRAM)

**Potential Issues:**
- If latency increase $> 2x$: impractical for real-time applications
- If verification requires $> 500$ ms: too slow for interactive use

**Follow-up:** Optimize representation extraction, use amortized verification (batch processing)

---

### Q5.2: Benign Performance Degradation

**Empirical Question:** How much does causal training hurt performance on legitimate tasks?

**Measurement Approach:**
- Evaluate on standard benchmarks: MMLU, HumanEval, TruthfulQA, etc.
- Compare baseline vs. causal model
- Measure degradation on each task category

**Success Criterion:**
- Average degradation $< 2\%$ across benchmarks
- No single benchmark degrades by $> 5\%$
- Critical tasks (e.g., safety-critical Q&A) degrade by $< 1\%$

**Potential Issues:**
- If degradation $> 5\%$ average: trade-off too costly
- If some tasks degrade by $> 20\%$: causal constraint breaks functionality

**Follow-up:** Use multi-objective optimization to balance robustness and performance; identify which tasks are most affected and why

---

### Q5.3: Continual Learning

**Empirical Question:** Can the model adapt to new attack types without catastrophic forgetting?

**Measurement Approach:**
- Train initial model on attacks $\mathcal{F}_1, \mathcal{F}_2$
- Deploy and monitor for new attack $\mathcal{F}_3$
- Retrain on $\mathcal{F}_3$, measure performance on $\mathcal{F}_1, \mathcal{F}_2$ (forgetting)
- Test continual learning methods: EWC, replay buffer, etc.

**Success Criterion:**
- After retraining, original attack success increases by $< 3\%$ (minimal forgetting)
- New attack success decreases to $< 10\%$
- Benign performance remains within $1\%$ of baseline

**Potential Issues:**
- If original attack success increases by $> 10\%$: catastrophic forgetting
- If new attack cannot be learned without hurting old: plasticity-stability dilemma

**Follow-up:** Use mixture-of-experts architecture, or periodic full retraining on combined dataset

---

## Category 6: Theoretical Extensions

These questions explore extensions beyond the current framework.

### Q6.1: Counterfactual Robustness

**Theoretical Question:** Can we extend from Level 2 (intervention) to Level 3 (counterfactual) reasoning?

**Definition:** Counterfactual robustness: "If the user input had not contained an injection, would the output have been the same?"

**Formalization:** $P(O_{u'} = o \mid U = u, O = o')$ where $u' = (u_{\text{data}}, \emptyset)$

**Empirical Question:** Does counterfactual reasoning provide stronger guarantees than interventional?

**Measurement Approach:**
- Implement counterfactual inference using structural equations
- Compare: interventional robustness vs. counterfactual robustness
- Measure: does counterfactual analysis identify attacks missed by interventional approach?

**Success Criterion:**
- Counterfactual method detects $> 10\%$ more attacks than interventional
- Provides interpretable explanations: "Output would have been X if injection removed"

**Potential Issues:**
- If counterfactual = interventional: no added value (use simpler interventional approach)
- If counterfactual inference is intractable: computational cost prohibitive

**Follow-up:** If beneficial, develop efficient counterfactual inference algorithms; extend theory to Level 3

---

### Q6.2: Multi-Turn Dialogue

**Theoretical Question:** How to extend SCM to multi-turn interactions?

**Challenge:** In dialogue, $O_t$ influences $U_{t+1}$, creating feedback loops that violate acyclicity.

**Proposed Extension:** Dynamic Bayesian Network with time-indexed variables:
$$S_t \to R_t \leftarrow U_t, \quad R_t \to O_t, \quad O_t \to U_{t+1}$$

**Empirical Question:** Do multi-turn attacks exploit temporal dependencies not captured by single-turn model?

**Measurement Approach:**
- Collect multi-turn attack dataset (e.g., iterative jailbreaks)
- Train single-turn causal model, test on multi-turn attacks
- Compare to multi-turn causal model with temporal structure
- Measure attack success difference

**Success Criterion:**
- Single-turn model achieves $> 70\%$ of multi-turn model's robustness
- Multi-turn model reduces attack success by $> 20\%$ compared to single-turn

**Potential Issues:**
- If single-turn model completely fails ($> 80\%$ attack success on multi-turn): need temporal modeling
- If multi-turn modeling is complex and provides $< 10\%$ improvement: not worth the cost

**Follow-up:** Develop full dynamic causal model for dialogue; extend all theorems to temporal setting

---

### Q6.3: Causal Explanation for Interpretability

**Empirical Question:** Can causal reasoning provide human-interpretable explanations for model behavior?

**Proposed Explanations:**
- "Output follows system instruction $S$ because representation $R$ blocks instruction content from user input $U$"
- "Attack failed because $U_{\text{instr}}$ was d-separated from $R$ by $S$"
- "Output changed because data content $U_{\text{data}}$ changed, not because of injected instructions"

**Measurement Approach:**
- Generate causal explanations for 100 test cases
- Present to human evaluators: baseline model explanations vs. causal explanations
- Measure: comprehensibility (Likert scale 1-5), accuracy (agreement with ground truth), usefulness (helps identify attacks)

**Success Criterion:**
- Comprehensibility $> 4.0$ / 5 (understandable to non-experts)
- Accuracy $> 90\%$ (explanations correctly describe causal relationships)
- Usefulness: evaluators correctly identify attacks in $> 80\%$ of cases using explanations

**Potential Issues:**
- If comprehensibility $< 3.0$: too technical, not useful for end users
- If accuracy $< 70\%$: explanations misleading

**Follow-up:** Develop NLG system to translate causal reasoning into natural language; user study with diverse populations

---

## Priority Ranking for ISEF Timeline

Given 6-month timeline to ISEF (May 2025), prioritize as follows:

### Must-Have (Required for Publication)
1. **Q1.1:** D-separation validation - core theoretical claim
2. **Q2.1:** Causal estimation error measurement - practical certificate
3. **Q4.1:** Novel attack family transfer - main empirical contribution
4. **Q5.2:** Benign performance - must not break functionality

### Should-Have (Strengthens Paper)
5. **Q1.2:** Data preservation - ensures model works
6. **Q2.2:** PAC-Bayesian bound tightness - validates theory
7. **Q3.1:** Instruction identification method - practical implementation
8. **Q4.2:** Adaptive attacks - robustness to strong adversary

### Nice-to-Have (Extensions for Discussion)
9. **Q1.4:** Causal discovery validation - additional evidence
10. **Q5.1:** Computational overhead - deployment feasibility
11. **Q6.1:** Counterfactual robustness - future work teaser

### Future Work (Post-ISEF)
12. **Q6.2:** Multi-turn dialogue - major extension
13. **Q6.3:** Causal explanations - separate paper
14. Remaining questions - incremental improvements

---

## Timeline and Milestones

### Month 1 (October 2025)
- **Q3.1:** Develop instruction identification method
- **Q1.1:** Implement HSIC tests for d-separation
- **Q2.1:** Implement causal estimation error measurement

### Month 2 (November 2025)
- Train initial causal model
- **Q1.1, Q2.1:** Validate on training data
- **Q5.2:** Measure benign performance

### Month 3 (December 2025)
- **Q4.1:** Test on held-out novel attack families
- **Q2.2:** Compute PAC-Bayesian bound
- **Q1.2:** Validate data preservation

### Month 4 (January 2025)
- **Q4.2:** Red team with adaptive attacks
- **Q5.1:** Measure computational overhead
- Comprehensive ablation studies

### Month 5 (February 2025)
- Complete all Must-Have + Should-Have experiments
- Begin paper writing
- Prepare ISEF presentation materials

### Month 6 (March-April 2025)
- Final experiments and polish
- Paper submission draft complete
- Demo implementation for ISEF

---

## Success Metrics Summary

**Minimum Viable Results for ISEF:**
- $\epsilon_{\text{causal}} < 0.05$ (measurable robustness certificate)
- Attack success on novel families $< 10\%$ (strong generalization)
- Benign performance degradation $< 2\%$ (practical)
- D-separation validated: HSIC $< 0.05$ (theory confirmed)

**Strong Results for Publication:**
- $\epsilon_{\text{causal}} < 0.03$ (very strong robustness)
- Novel attack success $< 5\%$ (SOTA)
- PAC-Bayesian bound non-vacuous and within 2x of empirical (theory validated)
- Adaptive attacks increase success by $< 2x$ (robust to strong adversary)
- Works across $\geq 3$ model architectures (general)

**Outstanding Results for Top-Tier Venue:**
- $\epsilon_{\text{causal}} < 0.01$ (near-perfect robustness)
- Novel attack success $< 2\%$ (far beyond baselines)
- Theory perfectly predicts empirical performance ($R^2 > 0.95$)
- Zero-shot transfer to new domains (e.g., multimodal)
- Deployed system with runtime guarantees

---

**Next Steps:**
1. Begin with Q3.1 (instruction identification) - blocking for all other questions
2. Implement measurement infrastructure (HSIC, MMD, KL divergence estimation)
3. Generate counterfactual training data
4. Train first causal model
5. Systematically validate each must-have question

**Documentation:** Record all experimental results, failures, and insights. Maintain experiment log for reproducibility and paper writing.
