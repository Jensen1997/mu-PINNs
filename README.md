# Multi-Viscosity Physics-Informed Neural Networks ($\mu$-PINNs)


预测时输入命令：
python mupinns_trad.py --state predict --scenario cylinder --mu 1e-2 --load_model_dir ./model/model_trad_cylinder.pth

OpenFOAM数据说明：
进入OpenFOAM环境后，修改文件transportProperties（physicalProperties）中的粘度值，直接运行Allrun，即可获得对应粘度下数值模拟结果，结果存储在VTK文件夹中。
通过paraview进行可视化，同时可将结果导出为csv文件。

@article{zhang2023multi,
  title={Multi-Viscosity Physics-Informed Neural Networks for Generating Ultra High Resolution Flow Field Data},
  author={Zhang, Sen and Guo, Xiao-Wei and Li, Chao and Zhao, Ran and Yang, Canqun and Wang, Wei and Zhong, Yanxu},
  journal={International Journal of Computational Fluid Dynamics},
  pages={1--19},
  year={2023},
  publisher={Taylor \& Francis}
}