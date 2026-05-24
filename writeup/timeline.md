# 6-Week Timeline

A part-time research project, 5-8 hours per week. Adjust if you have more or less time.

## Week 0 (this week) — Warm up

**Goal:** Confirm environment works. Re-familiarize with PyTorch. Read the paper.

- [ ] Run `notebooks/00_pytorch_warmup.ipynb` end-to-end without errors
- [ ] Read the original paper (Li et al. 2018, arxiv:1712.09913) once cover-to-cover — don't worry about understanding every detail yet
- [ ] Skim the original repo at github.com/tomgoldstein/loss-landscape to see what code exists
- [ ] Push the project to GitHub. Public repo. **This is your portfolio piece — start it now, even half-done.**

**Time:** 4-6 hours

---

## Week 1 — Understand the math

**Goal:** Understand exactly what filter normalization is and why it's needed.

- [ ] Read the paper again, this time slowly. Focus on Section 4 (the filter normalization technique).
- [ ] Work through the math by hand: for a 3×3×3 conv filter, write out what filter normalization does to a random direction.
- [ ] Create `notebooks/01_understanding_the_paper.ipynb` with your written-out math, in LaTeX cells.
- [ ] In the same notebook, implement a tiny toy version of filter normalization on a 5-filter conv layer. Verify your implementation matches the paper's equation.
- [ ] Begin writing `writeup/notes.md` — a running document of what you learn.

**Time:** 5-7 hours

---

## Week 2 — Train both networks

**Goal:** Get trained ResNet-20 and PlainNet-20 models.

- [ ] Test `src/train.py` runs on your machine. Try 5 epochs first to confirm.
- [ ] Train ResNet-20 for the full 100 epochs (~1 hour on GPU, overnight on CPU)
- [ ] Train PlainNet-20 for the full 100 epochs
- [ ] Document final test accuracies in `writeup/notes.md` — expect roughly: ResNet-20 ~91%, PlainNet-20 ~85-88% (skip connections help generalization too)
- [ ] If you're on CPU and 100 epochs is too slow, use Google Colab (free GPU)

**Time:** 4-6 hours of active work + training time

---

## Week 3 — Compute the loss surface

**Goal:** Reproduce the central figure of the paper — the 2D loss surface visualization for both networks.

- [ ] Create `notebooks/02_compute_loss_surface.ipynb`
- [ ] Load your trained ResNet-20
- [ ] Sample two random directions; apply filter normalization
- [ ] Compute the loss on a 21×21 grid (smaller than the paper's 51×51 for speed)
- [ ] Save the result as `figures/resnet20_landscape.npy`
- [ ] Repeat for PlainNet-20
- [ ] First sanity-check plots — they don't need to be pretty yet

**Time:** 6-8 hours

---

## Week 4 — Make the visualizations beautiful

**Goal:** Produce publication-quality figures showing the difference between the two landscapes.

- [ ] Create `notebooks/03_visualizations.ipynb`
- [ ] 3D surface plots for both networks (use `src/visualize.py:plot_3d_surface`)
- [ ] 2D contour plots for both networks
- [ ] Side-by-side comparison figure (the headline figure for your writeup)
- [ ] Increase resolution to 51×51 for the final figures (this is slow — leave it running overnight)
- [ ] Save all figures as 300 DPI PNGs in `figures/`

**Time:** 5-7 hours

---

## Week 5 — Extension experiments

**Goal:** Add one extension that's not in the paper. This is what makes your replication a *replication study* rather than just a code rewrite.

- [ ] Pick ONE extension:
  - **Extension A (recommended):** Train ResNet-20 with `save_every_epoch: true`. Compute the loss landscape at epochs 1, 10, 30, 60, 100. Animate how the landscape evolves over training.
  - **Extension B:** Disable batch normalization. Train both networks again. Compare the resulting landscapes. Does the "smoothing" effect of skip connections still hold without BN?
  - **Extension C (harder):** Replace SGD with Adam. Train both networks. Compare landscape geometry across optimizers.
- [ ] Create `notebooks/04_extension.ipynb`
- [ ] Document what you found in `writeup/notes.md`

**Time:** 7-10 hours

---

## Week 6 — Write it up and ship it

**Goal:** A polished writeup that a hiring manager or professor can read in 10 minutes.

- [ ] Create `writeup/full-writeup.md` — 1500-2000 words, sections:
  - Abstract (150 words)
  - Introduction (300 words)
  - Method (300 words)
  - Results — replication (300 words + 2 figures)
  - Results — extension (300 words + 1-2 figures)
  - Discussion (200 words)
  - Limitations & honest caveats (150 words)
  - References
- [ ] Polish all notebooks. Make sure they run end-to-end on a fresh clone.
- [ ] Update README with final results table (test accuracies, landscape metrics)
- [ ] Tag GitHub release v1.0
- [ ] Write a LinkedIn launch post linking to the repo
- [ ] Update your resume with the new bullet (see `writeup/resume-bullet.md`)
- [ ] Email 3 researchers whose work touches optimization or loss landscape geometry with a link

**Time:** 6-8 hours

---

## Total time investment

About 35-50 hours over 6 weeks, or 6-8 hours/week.

## What you have at the end

- A public GitHub repository with clean, reproducible code
- A 1500-2000 word writeup
- 5-8 publication-quality figures
- A 2-line resume bullet that signals serious research capability
- A real artifact to point to in any interview

## A note on honesty

If something doesn't work, document it. "I couldn't reproduce the result because X" is a valuable contribution. "I reproduced the result with deviations Y and Z" is also valuable. A failed replication, openly documented, is more impressive than a successful replication that hides its struggles.
