# Reliable Mammography Classification Framework Incorporating Wasserstein-Based Distributionally Robust Stochastic Optimization

This framework introduces a distributionally robust mammography classification approach that integrates a hierarchical Swin Transformer with Wasserstein-metric optimization. The methodology systematically addresses the inherent variability in medical imaging through three key mechanisms:

1. **Distribution Shift Handling**: Addresses domain variations by constraining adversarial perturbations within the Wasserstein ball, ensuring learned representations remain invariant to shifts across institutions and annotation styles
2. **Robust Optimization Against Label Noise**: Incorporates a Wasserstein-constrained adversarial training procedure to iteratively update input samples, effectively countering label noise and annotation inconsistencies
3. **Adversarial Defense in Multi-Institutional Deployments**: Ensures adversarial examples remain within a controlled Wasserstein distance, providing resilience against perturbations in real-world, multi-institutional settings

## Technical Architecture

The framework formulates the classification task as a minimax problem:

$$\min_{\theta} \max_{P \in B(\hat{P}, \epsilon)} \mathbb{E}_{(x,y) \sim P} [\ell(\theta; x, y)] ,$$

with $B(\hat{P}, \epsilon)$ representing a Wasserstein $\epsilon$-ball around the empirical distribution $\hat{P}$.

The optimization process employs an adaptive step size:

$$x_{\text{adv}}^{(t+1)} = x_{\text{adv}}^{(t)} + \alpha_t \left( \nabla_{x_{\text{adv}}} \ell(\theta; x_{\text{adv}}^{(t)}, y) - \beta(x_{\text{adv}}^{(t)} - x) \right),$$

with an adaptive step size $\alpha_t = \epsilon/\sqrt{t + 2}$ that enforces the Wasserstein constraint while effectively countering label noise and annotation inconsistencies.

### Theoretical Guarantees

This implementation provides several theoretical guarantees:

1. **Convergence Rate**: The algorithm converges at rate $O(1/\sqrt{T})$ to a local maximum of the adversarial loss.
2. **Wasserstein Constraint**: The final adversarial example satisfies: $W_2(x_{\text{adv}}^{(T)}, x) \leq \epsilon + O(1/\beta)$
3. **Loss Improvement**: The adversarial loss satisfies: $\ell(\theta; x_{\text{adv}}^{(T)}, y) \geq \ell(\theta; x, y) + \Omega(\epsilon^2)$ when $\beta$ is appropriately chosen.

This formulation ensures that the generated adversarial examples:
- Meaningfully increase the classification loss
- Remain within a controlled Wasserstein distance from the original sample
- Converge to stable perturbations that reflect realistic distribution shifts
- Maintain medical image validity through appropriate constraint selection

### Key Components

- **Architecture**: Hierarchical Swin Transformer adapted for single-channel mammogram processing
- **WRM Loss**: Implements Wasserstein Risk Minimization with controlled adversarial examples
- **Visualization Tools**: Includes attention map visualization and Grad-CAM for model interpretability
- **Class-Balanced Training**: Handles inherent class imbalance in medical imaging datasets


## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--root_dir` | Required | Path to directory containing CBIS-DDSM CSV files |
| `--dicom_path` | Optional | Path to text file containing list of DICOM file paths |
| `--output_dir` | ./outputs | Directory to save checkpoints, results, and visualizations |
| `--cls_epochs` | 1 | Number of epochs for classification training |
| `--batch_size` | 4 | Batch size for data loaders |
| `--num_workers` | 2 | Number of worker processes for data loading |
| `--train_cls` | False | Flag to train classification model |
| `--WRM_train` | False | Flag to enable Wasserstein Risk Minimization training |
| `--visualize` | False | Flag to generate attention maps and Grad-CAM visualizations |
| `--log_mode` | console | Controls logging behavior: 'console' for interactive display, 'file' for saving to output directory |

## Example Usage

```bash
python script.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_path /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --cls_epochs 10 \
  --batch_size 4 \
  --num_workers 2 \
  --train_cls \
  --WRM_train \
  --visualize
