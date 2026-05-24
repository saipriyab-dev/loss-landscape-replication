# Resume Bullet for This Project

Use after you've completed the project (or use the "in progress" version after Week 2).

---

## Final version (after Week 6, when published)

### Independent Research Project: Loss Landscape Visualization

**For ML Engineer / Applied Scientist roles:**

> Reproduced and extended *Visualizing the Loss Landscape of Neural Nets* (Li et al., NeurIPS 2018) from scratch in modern PyTorch. Trained paired ResNet-20 and PlainNet-20 architectures on CIFAR-10 ([X]% and [Y]% test accuracy respectively); implemented filter-normalized random directions to produce 2D loss-surface visualizations; quantitatively demonstrated the smoothing effect of skip connections on the loss geometry. Added an original extension analyzing [Extension A: landscape evolution during training / B: BN-ablated landscape comparison / C: optimizer effect on landscape geometry]. Code, figures, and writeup public at github.com/[username]/loss-landscape-replication.

**For Data Science / Research roles:**

> Conducted independent replication study of a NeurIPS 2018 paper on neural network loss landscape geometry. Implemented filter normalization, 2D loss-surface estimation, and publication-quality 3D visualizations in PyTorch and NumPy. Validated the paper's central empirical claim through controlled comparison of ResNet-20 vs. PlainNet-20 architectures. Authored a 2,000-word paper-style writeup documenting methodology, results, limitations, and one original extension experiment. Repository: [GitHub link].

**For Teaching / Educator roles:**

> Authored a 2,000-word technical replication study and accompanying public Jupyter notebooks reproducing a NeurIPS 2018 paper on the geometry of neural network loss landscapes. Notebooks are pedagogically structured with step-by-step explanations of filter normalization, gradient-based loss-surface estimation, and high-dimensional visualization techniques. Used as a portfolio demonstration of mathematical and pedagogical communication.

---

## In-progress version (use after Week 2)

Once you have trained models but not yet the visualizations, you can already list this on your resume:

> Currently conducting an independent replication study of *Visualizing the Loss Landscape of Neural Nets* (Li et al., NeurIPS 2018). Implementation in modern PyTorch underway; ResNet-20 and PlainNet-20 baselines trained on CIFAR-10 to validation accuracy of [X]% and [Y]% respectively. Project repository: github.com/[username]/loss-landscape-replication.

---

## How to talk about it in interviews

**The 30-second pitch:**
> "I replicated a 2018 NeurIPS paper from scratch in modern PyTorch — Li et al.'s work on visualizing the loss landscape of neural networks. The paper shows that skip connections smooth the loss geometry, which is why deep ResNets are easier to train than plain networks. I implemented the filter normalization technique they introduced, reproduced their core figures comparing ResNet-20 to a plain network, and added an extension looking at how the landscape evolves during training. The interesting math here is mostly calculus — gradient evaluation over a 2D grid in parameter space, with filter-norm scaling to make cross-architecture comparison meaningful."

**Why this answer works:**
- Specific paper, specific year, specific result — shows you can read and engage with research
- Names the technical innovation (filter normalization) — shows you understood the contribution
- Mentions the extension — shows independent thinking
- Connects to the math you love — shows authentic interest
- Quantifies the work (trained models, reproduced figures) — shows you finished

**Likely follow-up questions and good answers:**

*"Why did you pick this paper?"*
> "I'm interested in optimization and the geometry of loss surfaces. This paper is foundational for understanding why some architectures train more easily than others, and the math sits in single-variable calculus territory — gradient evaluation, norm computation, surface estimation. It was a good match for my interests and skill level."

*"What was hardest about it?"*
> Honest answer here. Maybe: "Filter normalization was conceptually clear but the implementation was finicky — getting the per-filter scaling right for conv layers vs. linear layers vs. batch-norm layers took several attempts. I wrote validation tests against the paper's reference implementation to confirm correctness."

*"What did you find in your extension?"*
> Specific to whichever extension you ran. Be honest about results — including null results.

---

## Where to put this on your resume

Best placement: a dedicated section titled **"Independent Research"** above your Experience section. This signals to readers immediately that you have research output, before they see your job titles.

If you have multiple projects (the loss landscape replication, plus future ones), each gets its own bullet. Don't combine them.
