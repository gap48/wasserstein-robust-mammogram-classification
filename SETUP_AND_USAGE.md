# Setup and Usage Instructions

## Required Directory Structure

Before running the code, ensure the following directory structure:
```markdown
project_root/
├── script.py                                 # Main script
├── run_script.py                             # Optional: Colab execution script
├── manifest-ZkhPvrLo5216730872708713142/
│   └── CBIS-DDSM/                           # Dataset directory
│       ├── mass_case_description_train_set.csv
│       ├── mass_case_description_test_set.csv
│       ├── calc_case_description_train_set.csv
│       ├── calc_case_description_test_set.csv
│       └── full mammogram images/
│           ├── Mass-Training_P_00001_LEFT_CC/
│           ├── Mass-Training_P_00001_LEFT_MLO/
│           └── ...
├── dcm_files.txt                             # Optional: List of DICOM paths
└── outputs/                                  # Created automatically
├── checkpoints/
├── plots/
├── visualizations/
└── training_log.txt
```
## Training Modes

### WRM-based Classification Training

To train the classification model with adversarial robustness using WRM:

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
  --log_mode console\
```
###  Standard Classification Training

If you wish to perform classification training without adversarial (WRM) updates:
```bash
python script.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_path /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --cls_epochs 10 \
  --batch_size 4 \
  --num_workers 2 \
  --train_cls \
  --log_mode console
```

###  Visualization Only
To generate attention maps and Grad-CAM visualizations using a previously trained model:
```bash
python script.py \
  --root_dir /path/to/manifest-<ID> \
  --dicom_path /path/to/dcm_files.txt \
  --output_dir ./outputs \
  --visualize \
  --log_mode console
```
###  Output Structure
The code generates the following outputs in the specified output_dir:
```markdown
outputs/
├── checkpoints/
│   └── best_checkpoint_wrm.pth               # Best model weights (highest validation accuracy)
├── plots/
│   ├── val_clean_accuracy.png                # Validation accuracy on clean samples
│   ├── val_adv_accuracy.png                  # Validation accuracy on adversarial samples
│   ├── test_clean_accuracy.png               # Test accuracy on clean samples
│   └── test_adv_accuracy.png                 # Test accuracy on adversarial samples
├── visualizations/
│   ├── attention_map.png                     # Attention map visualization
│   └── gradcam_map.png                       # Grad-CAM visualization
└── training_log.txt                          # Training logs (if log_mode=file)
```
