# -*- encoding: utf-8 -*-
'''
@File         :   get_points_cylinder.py
@Time         :   2023/01/12 13:30:27
@Author       :   Sen Zhang 
@Contact      :   zhangsen19@nudt.edu.cn
@Description  :   None
'''

# here put the import lib
import os, platform, time, sys, argparse, logging, pathlib, shutil
import numpy as np
from pyDOE import lhs
import matplotlib.pyplot as plt

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Get-Points-Cylinder')
    parser.add_argument('--scenario', type=str, default='cylinder', choices=['cylinder'], help='scenarios')
    parser.add_argument('--diameter', type=float, default=0.1, help='cylinder diameter')
    parser.add_argument('--xmax', type=float, default=2, help='Max x coordinate')
    parser.add_argument('--xmin', type=float, default=0, help='Min x coordinate')
    parser.add_argument('--ymax', type=float, default=1, help='Max y coordinate')
    parser.add_argument('--ymin', type=float, default=0, help='Mmin y coordinate')
    parser.add_argument('--circle_x', type=float, default=0.5, help='x coordinate of circle')
    parser.add_argument('--circle_y', type=float, default=0.5, help='y coordinate of circle')
    parser.add_argument('--U_max', type=float, default=1.0, help='Max u velocity')
    return parser.parse_args()

def DelSrcPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''Delete collocation point within cylinder'''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]


if __name__ == "__main__":

    args = parse_args()

    '''CREATE DIR'''
    start_time = time.time()
    str_start_time = time.strftime('%Z-%Y-%m-%d-%H%M%S', time.localtime(start_time))
    running_platform = platform.node()
    file_dir = os.path.abspath(os.path.dirname(__file__))
    file_name = os.path.basename(__file__).split('.')[0]
    working_dir = pathlib.Path(file_dir).joinpath('data')
    working_dir.mkdir(exist_ok=True)
    working_dir = working_dir.joinpath(str_start_time)
    working_dir.mkdir(exist_ok=True)

    '''COPY FILES'''
    shutil.copyfile(__file__, str(working_dir)+'/'+os.path.basename(__file__))

    '''SET LOGGER'''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%H:%M:%S")
    file_handler = logging.FileHandler('%s/log_%s.txt' % (working_dir, file_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Start time: " + str_start_time)
    logger.info('Running platform: ' + running_platform)
    logger.info('File name: ' + file_name)
    logger.info('File directory: ' + file_dir)
    logger.info('Working directory: ' + str(working_dir))

    '''PARAMETERS'''
    logger.info(args)
    scenario = args.scenario
    D = args.diameter
    xmax = args.xmax
    xmin = args.xmin
    ymax = args.ymax
    ymin = args.ymin
    circle_x = args.circle_x
    circle_y = args.circle_y
    U_max = args.U_max
    
    lb = np.array([xmin, ymin])
    ub = np.array([xmax, ymax])

    '''UP and DOWN'''
    UP = [xmin, ymax] + [xmax-xmin, 0] * lhs(2, 400)
    DOWN = [xmin, ymin] + [xmax-xmin, 0] * lhs(2, 400)

    '''INLET and OUTLET'''
    INLET = [xmin, ymin] + [0, ymax-ymin] * lhs(2, 400)
    OUTLET = [xmax, ymin] + [0, ymax-ymin] * lhs(2, 400)

    '''CYLINDER'''
    theta_time = [0] + [2*np.pi]*lhs(1, 400)
    x_cir = np.multiply(D/2, np.cos(theta_time)) + circle_x
    y_cir = np.multiply(D/2, np.sin(theta_time)) + circle_y
    CIR = np.concatenate((x_cir, y_cir), 1)

    '''INNER'''
    INNER_globel = lb + (ub - lb) * lhs(2, 26000)
    INNER_cir_refine = [circle_x/2, circle_y/2] + [circle_x, circle_y] * lhs(2, 3500)
    INNER_refine = [circle_x/2, circle_y/2] + [3*circle_x, circle_y] * lhs(2, 6500)
    INNER = np.concatenate((INNER_globel, INNER_cir_refine, INNER_refine), 0)
    INNER = DelSrcPT(INNER, xc=circle_x, yc=circle_y, r=D/2)
    INNER = np.concatenate((INNER, CIR, UP, DOWN, INLET, OUTLET), 0)

    y_inb = INLET[:,1:2]
    u_inb = 4*U_max*y_inb*(ymax-y_inb)/(ymax**2)
    v_inb = np.zeros_like(u_inb)
    U0 = np.concatenate((u_inb, v_inb), 1)

    '''SUMMARY'''
    logger.info('UP shape: ' + str(UP.shape))
    logger.info('DOWN shape: ' + str(DOWN.shape))
    logger.info('INLET shape: ' + str(INLET.shape))
    logger.info('OUTLET shape: ' + str(OUTLET.shape))
    logger.info('CIR shape: ' + str(CIR.shape))
    logger.info('INNER shape: ' + str(INNER.shape))
    logger.info('U0 [u, v] max: ' + str(U0.max(axis=0)))
    logger.info('U0 [u, v] min: ' + str(U0.min(axis=0)))
    num_other = UP.shape[0]+DOWN.shape[0]+INLET.shape[0]+OUTLET.shape[0]+CIR.shape[0]
    num_inner = INNER.shape[0]
    num_all = num_inner + num_other
    logger.info('Inner points: ' + str(num_inner))
    logger.info('Other points: ' + str(num_other))
    logger.info('Total points: ' + str(num_all))

    '''STORE data'''
    filename_store = '/'+scenario+'_'+'points'
    np.savez(str(working_dir) + filename_store +'.npz', \
            INNER=INNER, UP=UP, DOWN=DOWN, INLET=INLET, OUTLET=OUTLET, CIR=CIR, U0=U0)

    '''Visualize training points'''
    fig = plt.figure(figsize=(10, 4), layout="constrained")
    gs = plt.GridSpec(1, 9)
    
    ax1 = fig.add_subplot(gs[:,0:2])
    ax1.scatter(U0[:, 0:1], INLET[:, 1:2], marker='o', alpha=0.7, s=1, color='red')
    ax1.set_title('init u velocity')
    ax1.set_xlabel('u_velocity (m/s)')
    ax1.set_ylabel('y (m)')

    ax2 = fig.add_subplot(gs[:,2:10])
    ax2.scatter(INNER[:, 0:1], INNER[:, 1:2], marker='.', alpha=0.1, s=1, color='green')
    ax2.scatter(UP[:, 0:1], UP[:, 1:2], marker='.', alpha=0.5, s=1, color='yellow')
    ax2.scatter(DOWN[:, 0:1], DOWN[:, 1:2], marker='.', alpha=0.5, s=1, color='yellow')
    ax2.scatter(CIR[:, 0:1], CIR[:,1:2], marker='.', alpha=0.5, s=1, color='orange')
    ax2.scatter(INLET[:, 0:1], INLET[:, 1:2], marker='.', alpha=0.5, s=1, color='red')
    ax2.scatter(OUTLET[:, 0:1], OUTLET[:, 1:2], marker='.', alpha=0.5, s=1, color='blue')
    ax2.set_aspect(1)
    ax2.set_adjustable('datalim')
    ax2.set_title('points')
    ax2.set_yticklabels([])
    ax2.set_xlabel('x (m)')
    fig.savefig(str(working_dir) + filename_store +'.png', dpi=300)
