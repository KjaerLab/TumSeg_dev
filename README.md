Demonstraion of how to use the command line interface for TumSeg. 

Installation:
```python
conda create -n tumseg_env python=3.9 
conda activate tumseg_env
cd <path to TumSeg folder>
pip install -e .
```

Basic usage:
```python
python predict.py -i /path/to/data -o /path/to/output/folder --run_uq
```
The input can be either the path to a nifti file ending with .nii or .nii.gz, or to a folder containing the nifti files. To run the UQ module for predicting the expected Dice score use the `--run_uq` flag.

Sample output:
```
Analyzing: /home/Study 1/M09_M10_22.5h/M10_22.5h/CT_M10_22.nii.gz
resampling..
Running Monte-Carlo samples for UQ..
100%|████████████████████████████████████████████████████████████| 100/100 

Expected Dice score: 0.976

Saving results to: /home/output_folder/Study 1/M09_M10_22.5h/M10_22.5h/CT_M10_22.nii.gz
Done.
```

If you use this work, please cite the following

```
Jensen, M., Clemmensen, A., Hansen, J.G., van Krimpen Mortensen, J., Christensen, E.N., Kjaer, A. and Ripa, R.S., 2024. 3D whole body preclinical micro-CT database of subcutaneous tumors in mice with annotations from 3 annotators. Scientific Data, 11(1), p.1021.
```


[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
