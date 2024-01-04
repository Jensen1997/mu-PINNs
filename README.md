# Multi-Viscosity Physics-Informed Neural Networks ($\mu$-PINNs)

This is a project about the article "Multi-Viscosity Physics-Informed Neural Networks for Generating Ultra High Resolution Flow Field Data". 

The project provides code and associated datasets to generate ultra-high resolution flow field data. You can use the provided code to generate training points, train a model, and make flow field predictions. We also provide specific examples using OpenFOAM where viscosity values can be modified to obtain corresponding numerical simulation results.

## File Structure

- `data/` folder: Contains the training points used during training and the data generated using OpenFOAM.
- `get_point_train/` folder: Contains code for generating training points.
- `model/` folder: Contains the trained model.
- `openfoam/` folder: Contains specific examples using OpenFOAM.

## Usage

### Prediction

Run the following command to make a prediction:

```
python mupinns_trad.py --state predict --scenario cylinder --mu 1e-2 --load_model_dir ./model/model_trad_cylinder.pth
```

### Using OpenFOAM

1. After entering the OpenFOAM environment, modify the viscosity value in the `transportProperties` (or `physicalProperties`) file.
2. Run the `Allrun` script to obtain the numerical simulation results for the corresponding viscosity value, which will be saved in the `VTK` folder.
3. Use Paraview for visualization and to export the results as a CSV file.

## Citation

If you find this project helpful, please cite our paper as follows:

```
@article{zhang2023multi,
  title={Multi-Viscosity Physics-Informed Neural Networks for Generating Ultra High Resolution Flow Field Data},
  author={Zhang, Sen and Guo, Xiao-Wei and Li, Chao and Zhao, Ran and Yang, Canqun and Wang, Wei and Zhong, Yanxu},
  journal={International Journal of Computational Fluid Dynamics},
  pages={1--19},
  year={2023},
  publisher={Taylor \& Francis}
}
```

## Contact Information

If you have any questions or suggestions, feel free to contact the project authors.