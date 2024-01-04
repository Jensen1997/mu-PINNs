# -*- encoding: utf-8 -*-
'''
@File         :   get_points_stepflow.py
@Time         :   2023/01/13 20:08:32
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
    parser = argparse.ArgumentParser('Get-Points-StepFlow')
    parser.add_argument('--scenario', type=str, default='stepflow', choices=['stepflow'], help='scenarios')
    parser.add_argument('--xmax', type=float, default=2, help='Max x coordinate')
    parser.add_argument('--xmin', type=float, default=0, help='Min x coordinate')
    parser.add_argument('--ymax', type=float, default=1, help='Max y coordinate')
    parser.add_argument('--ymin', type=float, default=0, help='Min y coordinate')
    parser.add_argument('--flength', type=float, default=0.95, help='Forward Length of step')
    parser.add_argument('--blength', type=float, default=0.95, help='Backward Length of step')
    parser.add_argument('--height', type=float, default=0.5, help='Height of step')
    parser.add_argument('--U_max', type=float, default=1.0, help='Max u velocity')
    return parser.parse_args()

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
    shutil.copyfile(__file__, str(working_dir)+'/0_'+os.path.basename(__file__))

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
    xmax = args.xmax
    xmin = args.xmin
    ymax = args.ymax
    ymin = args.ymin
    flength = args.flength
    blength = args.blength
    height = args.height
    U_max = args.U_max
    
    lb = np.array([xmin, ymin])
    ub = np.array([xmax, ymax])

    if (flength == 0 and blength == 0): 
        logger.error('flength and blength cannot be zero at the same time.')
        sys.exit(1)
    if flength == 0:
        logger.info('backward step')

    elif blength == 0:
        logger.info('forward step')
        scenario = scenario+'Forward'

        '''UP and DOWN'''
        UP = [xmin, ymax] + [xmax-xmin, 0] * lhs(2, 500)
        Down_1 = [xmin, ymin] + [flength, 0] * lhs(2, 250)
        Down_2 = [xmin+flength, ymin] + [0, height] * lhs(2, 200)
        Down_3 = [xmin+flength, ymin+height] + [xmax-xmin-flength, 0] * lhs(2, 250)
        DOWN = np.concatenate((Down_1, Down_2, Down_3), 0)

        '''INLET and OUTLET'''
        INLET = [xmin, ymin] + [0, ymax-ymin] * lhs(2, 400)
        OUTLET = [xmax, ymin+height] + [0, ymax-ymin-height] * lhs(2, 400)

        '''INNER'''
        Inner_1 = [xmin, ymin] + [flength, ymax-ymin] * lhs(2, 16000)
        Inner_2 = [xmin+flength, ymin+height] + [xmax-xmin-flength, ymax-ymin-height] * lhs(2, 8000)
        Inner_1_refine = [xmin+flength, ymin] + [-0.3, height+0.3] * lhs(2, 4500)
        Inner_2_refine = [xmax, ymin+height] + [-(xmax-xmin-flength+0.3), 0.3] * lhs(2, 7500)
        INNER = np.concatenate((Inner_1, Inner_2, Inner_1_refine, Inner_2_refine), 0)
        INNER = np.concatenate((INNER, UP, DOWN, INLET, OUTLET), 0)

        y_inb = INLET[:,1:2]
        u_inb = 4*U_max*y_inb*(ymax-y_inb)/(ymax**2)
        v_inb = np.zeros_like(u_inb)
        U0 = np.concatenate((u_inb, v_inb), 1)

    else:
        logger.info('column step')
        scenario = scenario+'Flap'

        '''UP and DOWN'''
        UP = [xmin, ymax] + [xmax-xmin, 0] * lhs(2, 500)
        Down_1 = [xmin, ymin] + [flength, 0] * lhs(2, 120)
        Down_2 = [xmin+flength, ymin] + [0, height] * lhs(2, 120)
        Down_3 = [xmin+flength, ymin+height] + [xmax-xmin-flength-blength, 0] * lhs(2, 20)
        Down_4 = [xmax-blength, ymin] + [0, height] * lhs(2, 120)
        Down_5 = [xmax-blength, ymin] + [blength, 0] * lhs(2, 120)
        DOWN = np.concatenate((Down_1, Down_2, Down_3, Down_4, Down_5), 0)

        '''INLET and OUTLET'''
        INLET = [xmin, ymin] + [0, ymax-ymin] * lhs(2, 500)
        OUTLET = [xmax, ymin] + [0, ymax-ymin] * lhs(2, 500)

        '''INNER'''
        Inner_1 = [xmin, ymin] + [flength, ymax-ymin] * lhs(2, 16000)
        Inner_2 = [xmin+flength, ymin+height] + [xmax-xmin-flength-blength, ymax-ymin-height] * lhs(2, 1000)
        Inner_3 = [xmax-blength, ymin] + [blength, ymax-ymin] * lhs(2, 16000)
        Inner_1_refine = [xmin+flength, ymin] + [-0.3, height+0.3] * lhs(2, 1400)
        Inner_2_refine = [xmin+flength-0.3, ymin+height] + [xmax-xmin-flength-blength+0.6, 0.3] * lhs(2, 200)
        Inner_3_refine = [xmax-blength, ymin] + [0.3, height+0.3] * lhs(2, 1400)
        INNER = np.concatenate((Inner_1, Inner_2, Inner_3, Inner_1_refine, Inner_2_refine, Inner_3_refine), 0)
        INNER = np.concatenate((INNER, UP, DOWN, INLET, OUTLET), 0)

        y_inb = INLET[:,1:2]
        u_inb = 4*U_max*y_inb*(ymax-y_inb)/(ymax**2)
        v_inb = np.zeros_like(u_inb)
        U0 = np.concatenate((u_inb, v_inb), 1)


        logger.info('building')

    '''SUMMARY'''
    logger.info('UP shape: ' + str(UP.shape))
    logger.info('DOWN shape: ' + str(DOWN.shape))
    logger.info('INLET shape: ' + str(INLET.shape))
    logger.info('OUTLET shape: ' + str(OUTLET.shape))
    logger.info('INNER shape: ' + str(INNER.shape))
    logger.info('U0 [u, v] max: ' + str(U0.max(axis=0)))
    logger.info('U0 [u, v] min: ' + str(U0.min(axis=0)))
    num_other = UP.shape[0]+DOWN.shape[0]+INLET.shape[0]+OUTLET.shape[0]
    num_inner = INNER.shape[0]
    num_all = num_inner + num_other
    logger.info('Inner points: ' + str(num_inner))
    logger.info('Other points: ' + str(num_other))
    logger.info('Total points: ' + str(num_all))

    '''STORE data'''
    filename_store = '/'+scenario+'_'+'points'
    np.savez(str(working_dir) + filename_store +'.npz', \
            INNER=INNER, UP=UP, DOWN=DOWN, INLET=INLET, OUTLET=OUTLET, U0=U0)

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
    ax2.scatter(INLET[:, 0:1], INLET[:, 1:2], marker='.', alpha=0.5, s=1, color='red')
    ax2.scatter(OUTLET[:, 0:1], OUTLET[:, 1:2], marker='.', alpha=0.5, s=1, color='blue')
    ax2.set_aspect(1)
    ax2.set_adjustable('datalim')
    ax2.set_title('points')
    ax2.set_yticklabels([])
    ax2.set_xlabel('x (m)')
    fig.savefig(str(working_dir) + filename_store +'.png', dpi=300)
