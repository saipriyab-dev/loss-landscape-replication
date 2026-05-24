# Loss Landscape Visualization — A Modern Replication Study

A from-scratch PyTorch reimplementation and extension of:

> **Visualizing the Loss Landscape of Neural Nets**
> Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein
> NeurIPS 2018 · [arXiv:1712.09913](https://arxiv.org/abs/1712.09913)

## Motivation

Why do skip connections in ResNets make training easier? Why are some loss landscapes "well-behaved" and others chaotic? The 2018 Li et al. paper proposed a method called **filter normalization** for visualizing the loss surface of a trained network in a way that makes meaningful comparisons across architectures possible. The visualizations are now standard pedagogical material in deep learning, but the original codebase is over six years old and uses deprecated PyTorch APIs.

This project reimplements the core method from scratch in modern PyTorch (≥ 2.0), validates the central empirical claim — that skip connections smooth the loss landscape — and adds two extensions the original paper did not explore.

## Research questions

1. **Replication:** Can the qualitative finding "skip connections smooth the loss landscape" be reproduced with modern training pipelines on CIFAR-10?
2. **Extension A:** How does the loss landscape evolve over training? The original paper visualizes only the final landscape. Does the geometry change as training progresses?
3. **Extension B:** How does batch normalization interact with filter normalization in the visualization? The original paper assumes BN but doesn't ablate.

## Method summary

The filter normalization technique (single-variable calculus made geometric):

1. Train two networks: one with skip connections (ResNet-20), one without (a "PlainNet-20" of equivalent depth).
2. For each trained network, sample two random direction vectors `δ` and `η` in parameter space.
3. **Normalize each direction per-filter** so the magnitude of the perturbation is scaled to the magnitude of the corresponding filter — this is the key innovation that makes cross-architecture comparison meaningful.
4. Evaluate `L(θ* + α·δ + β·η)` on a grid of `(α, β)` values to produce a 2D loss surface, where `θ*` are the trained parameters.
5. Plot as a 3D surface and 2D contour map.

The mathematics is mostly: gradient evaluation, norm computation, and surface plotting — all single-variable calculus extended to high-dimensional parameter space.

## Results (to be filled in)

| Architecture | Test accuracy | Landscape minimum width (η_50) | Filter-norm ratio |
|---|---|---|---|
| ResNet-20 | TBD | TBD | TBD |
| PlainNet-20 | TBD | TBD | TBD |

Final figures will be linked from `figures/`.

## Repository layout

```
loss-landscape-replication/
├── notebooks/          # Numbered analysis notebooks (00-05)
├── src/                # Reusable modules (models, training, visualization)
├── configs/            # YAML training configurations
├── data/               # CIFAR-10 (auto-downloaded, gitignored)
├── models/             # Trained model checkpoints (gitignored)
├── figures/            # Final publication figures
├── writeup/            # Final paper-style writeup
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Reproducing the results

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/loss-landscape-replication.git
cd loss-landscape-replication
pip install -r requirements.txt

# 2. Train the two networks (~2 hours on a single GPU, longer on CPU)
python src/train.py --config configs/resnet20.yaml
python src/train.py --config configs/plainnet20.yaml

# 3. Generate the loss landscape visualizations
python src/visualize.py --model models/resnet20_final.pt --output figures/resnet20_landscape.png
python src/visualize.py --model models/plainnet20_final.pt --output figures/plainnet20_landscape.png

# 4. Walk through the notebooks for the full analysis
jupyter lab notebooks/
```

## Timeline

This is a 6-week part-time research project. See `writeup/timeline.md` for weekly milestones.

## References

- Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018). *Visualizing the Loss Landscape of Neural Nets.* NeurIPS 2018.
- Original implementation: [github.com/tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape) (referenced for correctness, not copied)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.

## About this project

This is an independent research project conducted as part of a public portfolio. The author has a background in software engineering and data analytics and is studying numerical methods, optimization, and the mathematical foundations of deep learning.

Author: Saipriya Bethi
Location: Dallas, TX
Contact: saipriyab.3009@gmail.com

## License

MIT. Use freely; attribution appreciated.
