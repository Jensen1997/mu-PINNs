# -*- encoding: utf-8 -*-
'''
@File         :   mupinns_mixed.py
@Time         :   2023/03/15 15:12:50
@Author       :   Sen Zhang
@Contact      :   zhangsen19@nudt.edu.cn
@Description  :   Mixed-Variable Scheme
'''

# here put the import lib

import os, platform, time, sys, argparse, logging, pathlib, shutil
import numpy as np
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import csv

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('mu-pinns')
    parser.add_argument('--state', type=str, default='train', choices=['train', 'predict'], help='running state')
    parser.add_argument('--scenario', type=str, default='cylinder', choices=['cylinder', 'stepflowFlap', 'stepflowForward'], help='scenarios')
    parser.add_argument('--rho', type=float, default=1.0, help='density')
    parser.add_argument('--mu', type=str, default='1e-2', help='dynamic viscosity')
    parser.add_argument('--load_model_dir', type=str, default='./model.pth', help='model directory for loading')
    parser.add_argument('--num_inner_batch_size', type=int, default=18000, help='inner batch size in training')
    return parser.parse_args()

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, Setting, INNER, INLET, OUTLET, WALL, U0):
        super(NeuralNetwork, self).__init__()
        
        self.in_dim = 3
        self.out_dim = 6 # Mixed-Variable Scheme
        self.tanh = nn.Tanh()
        # uv net
        self.uvn_fc1 = nn.Linear(self.in_dim, 64)
        self.uvn_fc2 = nn.Linear(64, 128)
        self.uvn_fc3 = nn.Linear(128, 128)
        self.uvn_fc4 = nn.Linear(128, 128)
        self.uvn_fc5 = nn.Linear(128, 128)
        self.uvn_fc6 = nn.Linear(128, 128)
        self.uvn_fc7 = nn.Linear(128, 64)
        self.uvn_fc8 = nn.Linear(64, self.out_dim)

        self.loss_fn = nn.MSELoss(reduction='mean')

        # Mat. properties & settings
        self.working_dir = Setting['working_dir']
        self.rho = Setting['rho']
        self.mu = Setting['mu']
        self.mu_min = np.log10(0.001)
        self.mu_max = np.log10(0.01)
        self.lb = torch.tensor([Setting['xmin'], Setting['ymin'], self.mu_min]).to(device)
        self.ub = torch.tensor([Setting['xmax'], Setting['ymax'], self.mu_max]).to(device)
        
        # Collocation point
        x_inner = torch.tensor(INNER[:, 0:1])
        y_inner = torch.tensor(INNER[:, 1:2])
        self.INNER_dataset = Data.TensorDataset(x_inner, y_inner)
        self.x = torch.tensor(INNER[:, 0:1], requires_grad=True).to(device)
        self.y = torch.tensor(INNER[:, 1:2], requires_grad=True).to(device)

        self.x_INLET = torch.tensor(INLET[:, 0:1], requires_grad=True).to(device)
        self.y_INLET = torch.tensor(INLET[:, 1:2], requires_grad=True).to(device)
        self.u_INLET = torch.tensor(U0[:, 0:1]).to(device)
        self.v_INLET = torch.tensor(U0[:, 1:2]).to(device)

        self.x_OUTLET = torch.tensor(OUTLET[:, 0:1], requires_grad=True).to(device)
        self.y_OUTLET = torch.tensor(OUTLET[:, 1:2], requires_grad=True).to(device)

        self.x_WALL = torch.tensor(WALL[:, 0:1], requires_grad=True).to(device)
        self.y_WALL = torch.tensor(WALL[:, 1:2], requires_grad=True).to(device)

    def forward(self, x, y, mu):
        Y = self.net_uv(x, y, mu)
        return Y

    def gradients(self, u, x):
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    def net_uv(self, x, y, mu):
        
        mu = torch.full_like(x,  torch.log10(mu)[0]).to(device)
        xy = torch.cat((x, y, mu), 1)
        X = 2.0*(xy - self.lb)/(self.ub - self.lb) - 1.0

        X = self.tanh(self.uvn_fc1(X))
        X = self.tanh(self.uvn_fc2(X))
        X = self.tanh(self.uvn_fc3(X))
        X = self.tanh(self.uvn_fc4(X))
        X = self.tanh(self.uvn_fc5(X))
        X = self.tanh(self.uvn_fc6(X))
        X = self.tanh(self.uvn_fc7(X))
        Y = self.uvn_fc8(X)

        return Y

    def net_f(self, x, y, mu):

        rho=self.rho

        Y = self.net_uv(x, y, mu)

        u   = Y[:, 0:1]
        v   = Y[:, 1:2]
        p   = Y[:, 2:3]
        s11 = Y[:, 3:4]
        s22 = Y[:, 4:5]
        s12 = Y[:, 5:6]

        s11_1 = self.gradients(s11, x)
        s12_2 = self.gradients(s12, y)
        s22_2 = self.gradients(s22, y)
        s12_1 = self.gradients(s12, x)

        u_x = self.gradients(u, x)
        u_y = self.gradients(u, y)

        v_x = self.gradients(v, x)
        v_y = self.gradients(v, y)

        # Eq: \nabla \cdot (\rho \mathbf{u}) = 0
        f_c = rho*(u_x + v_y)

        # Eq: \rho (\mathbf{u} \cdot \nabla)\mathbf{u} = \nabla \boldsymbol{\sigma}
        f_u = rho*(u*u_x + v*u_y) - s11_1 - s12_2
        f_v = rho*(u*v_x + v*v_y) - s12_1 - s22_2

        # Eq: \boldsymbol{\sigma} = -p \mathbf{I} + \mu (\nabla \mathbf{u} + \nabla \mathbf{u}^T)
        f_s11 = -p + 2*mu*u_x - s11
        f_s22 = -p + 2*mu*v_y - s22
        f_s12 = mu*(u_y+v_x) - s12

        # Eq: p = \frac{1}{2} \sum_{i=1}^{n} \sigma_{ii}
        f_p = p + (s11+s22)/2

        return f_c, f_u, f_v, f_s11, f_s22, f_s12, f_p

    def loss_calculate(self, x, y, mu):

        f_pred_c, f_pred_u, f_pred_v, f_pred_s11, f_pred_s22, f_pred_s12, f_pred_p = self.net_f(x, y, mu)
        Y_INLET_pred = self.net_uv(self.x_INLET, self.y_INLET, mu)
        Y_OUTLET_pred = self.net_uv(self.x_OUTLET, self.y_OUTLET, mu)
        Y_WALL_pred = self.net_uv(self.x_WALL, self.y_WALL, mu)
        u_INLET_pred = Y_INLET_pred[:,0:1]
        v_INLET_pred = Y_INLET_pred[:,1:2]
        p_OUTLET_pred = Y_OUTLET_pred[:,2:3]
        u_WALL_pred = Y_WALL_pred[:,0:1]
        v_WALL_pred = Y_WALL_pred[:,1:2]

        loss_f = self.loss_fn(f_pred_c, torch.zeros_like(f_pred_c)) + \
                 self.loss_fn(f_pred_u, torch.zeros_like(f_pred_u)) + \
                 self.loss_fn(f_pred_v, torch.zeros_like(f_pred_v)) + \
                 self.loss_fn(f_pred_s11, torch.zeros_like(f_pred_s11)) + \
                 self.loss_fn(f_pred_s22, torch.zeros_like(f_pred_s22)) + \
                 self.loss_fn(f_pred_s12, torch.zeros_like(f_pred_s12)) + \
                 self.loss_fn(f_pred_p, torch.zeros_like(f_pred_p))
        
        loss_WALL = self.loss_fn(u_WALL_pred, torch.zeros_like(u_WALL_pred)) + \
                    self.loss_fn(v_WALL_pred, torch.zeros_like(v_WALL_pred))

        loss_INLET = self.loss_fn(u_INLET_pred, self.u_INLET) + \
                     self.loss_fn(v_INLET_pred, self.v_INLET)

        loss_OUTLET = self.loss_fn(p_OUTLET_pred, torch.zeros_like(p_OUTLET_pred))

        loss = 10*loss_f + 50*loss_WALL + 50*loss_INLET + 50*loss_OUTLET

        return loss, loss_f, loss_WALL, loss_INLET, loss_OUTLET

    def train_Adam(self, epochs, learning_rate, start_time, logger, epoch0=0, writer_dir=0, write_interval=100, label_train_print=True):
        optimizer_Adam = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch_index in range(epoch0, epoch0+epochs):
            mu = 10**((self.mu_max-self.mu_min)*torch.rand(1)+self.mu_min).to(device)
            if label_train_print: print(f"\n-------------------------------\nAdam: Epoch {epoch_index+1} / {epoch0+epochs} with lr: {learning_rate}")
            self.train()
            self.zero_grad()

            loss, _, _, _, _ = self.loss_calculate(self.x, self.y, mu)
            # Back propagation
            optimizer_Adam.zero_grad()
            loss.backward()
            if label_train_print: print(f"loss: {loss.item():>8f}   mu: {mu[0]:>8f}")
            optimizer_Adam.step()

            if ((epoch_index+1)%write_interval==0 and writer_dir != 0):

                loss, loss_f, loss_WALL, loss_INLET, loss_OUTLET = self.loss_calculate(self.x, self.y, mu)
                logger.info(f"Adam: Epoch {epoch_index+1} / {epoch0+epochs} with lr: {learning_rate}")
                logger.info(f"loss: {loss.item():>8f}   mu: {mu[0]:>8f}")
                logger.info('Training time: %.1f s' % (time.time() - start_time))
                with open(writer_dir, 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    data_loss = [epoch_index+1, loss.item(), loss_f.item(), loss_WALL.item(), loss_INLET.item(), loss_OUTLET.item()]
                    writer.writerow(data_loss)

        model_dir = self.working_dir + '/model_Adam_' + str(epoch0+epochs) + '.pth'
        torch.save(model.state_dict(), model_dir)
        logger.info("Saved PyTorch Model State to \"" + model_dir + "\"")

        return epoch0+epochs


    def train_batch_Adam(self, epochs, learning_rate, inner_batch_size, start_time, logger, epoch0=0, writer_dir=0, write_interval=100, label_train_print=True):
        optimizer_Adam = torch.optim.Adam(self.parameters(), lr=learning_rate)
        Inner_dataloader = Data.DataLoader(self.INNER_dataset, inner_batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        num_batches = len(Inner_dataloader)
        
        for epoch_index in range(epoch0, epoch0+epochs):
            if label_train_print: print(f"\n-------------------------------\nAdam: Epoch {epoch_index+1} / {epoch0+epochs} with lr: {learning_rate}")
            self.train()
            self.zero_grad()

            for batch, (x, y) in enumerate(Inner_dataloader):
                x, y = x.to(device), y.to(device)
                x.requires_grad = True; y.requires_grad = True
                mu = 10**((self.mu_max-self.mu_min)*torch.rand(1)+self.mu_min).to(device)

                loss, _, _, _, _ = self.loss_calculate(x, y, mu)
                # Back propagation
                optimizer_Adam.zero_grad()
                loss.backward()
                if label_train_print: print(f"loss: {loss.item():>8f}    [{batch+1:>5d}/{num_batches:>5d}]  mu: {mu[0]:>8f}")
                optimizer_Adam.step()

            if ((epoch_index+1)%write_interval==0 and writer_dir != 0):
                x, y= next(iter(Inner_dataloader))
                x, y = x.to(device), y.to(device)
                x.requires_grad = True; y.requires_grad = True

                loss, loss_f, loss_WALL, loss_INLET, loss_OUTLET  = self.loss_calculate(x, y, mu)
                logger.info(f"Adam: Epoch {epoch_index+1} / {epoch0+epochs} with lr: {learning_rate}")
                logger.info(f"loss: {loss.item():>8f}   mu: {mu[0]:>8f}")
                logger.info('Training time: %.1f s' % (time.time() - start_time))
                with open(writer_dir, 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    data_loss = [epoch_index+1, loss.item(), loss_f.item(), loss_WALL.item(), loss_INLET.item(), loss_OUTLET.item()]
                    writer.writerow(data_loss)

        model_dir = self.working_dir + '/model_Adam_' + str(epoch0+epochs) + '.pth'
        torch.save(model.state_dict(), model_dir)
        logger.info("Saved PyTorch Model State to \"" + model_dir + "\"")

        return epoch0+epochs

    

if __name__ == "__main__":

    args = parse_args()

    '''CREATE DIR'''
    start_time = time.time()
    str_start_time = time.strftime('%Z-%Y-%m-%d-%H%M%S', time.localtime(start_time))
    running_platform = platform.node()
    file_dir = os.path.abspath(os.path.dirname(__file__))
    file_name = os.path.basename(__file__).split('.')[0]
    working_dir = pathlib.Path(file_dir).joinpath(args.state)
    working_dir.mkdir(exist_ok=True)
    working_dir = working_dir.joinpath(str_start_time)
    working_dir.mkdir(exist_ok=True)

    '''SET LOGGER'''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt="%H:%M:%S")
    file_handler = logging.FileHandler('%s/log_%s_%s_mu%s.txt' % (working_dir, args.state, args.scenario, args.mu))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Start time: " + str_start_time)
    logger.info("Using {} device".format(device))
    logger.info('Running platform: ' + running_platform)
    logger.info('Running state: ' + args.state)
    logger.info('File directory: ' + file_dir)
    logger.info('File name: ' + file_name)
    logger.info('Working directory: ' + str(working_dir))

    '''PARAMETERS'''
    logger.info(args)

    # Domain bounds
    xmax = 2
    xmin = 0
    ymax = 1
    ymin = 0
    rho = args.rho
    mu = torch.tensor(float(args.mu)).view(-1)
        
    Setting = {'rho':rho, 'mu':mu, \
                'xmax':xmax, 'xmin':xmin,\
                'ymax':ymax, 'ymin':ymin,\
                'working_dir':str(working_dir)}
    logger.info(Setting)

    num_inner_batch_size = args.num_inner_batch_size
    data_dir = file_dir + f'/data/{args.scenario}_points.npz'
    of_data_dir = file_dir + f'/data/{args.scenario}/{args.scenario}_{args.mu}mu.npy'

    # prepare predict points
    logger.info("Loading OpenFOAM Data from \"" + of_data_dir + "\"")
    '''['Points_0', 'Points_1', 'U_0', 'U_1', 'U_Magnitude', 'p']'''
    XY_pred = np.load(of_data_dir)
    x_coordinate = XY_pred[:,0:1]
    y_coordinate = XY_pred[:,1:2]
    u_exact = XY_pred[:,2:3]
    v_exact = XY_pred[:,3:4]
    uv_mag_exact = XY_pred[:,4:5]
    p_exact = XY_pred[:,5:6]
    x_pred_tensor = torch.tensor(x_coordinate, requires_grad=True).to(device)
    y_pred_tensor = torch.tensor(y_coordinate, requires_grad=True).to(device)
    
    # loading data & model
    logger.info("Loading data from \""+data_dir+"\"")
    points = np.load(data_dir)
    if 'cylinder' in args.scenario:
        INNER = points['INNER']
        INLET = points['INLET']
        OUTLET = points['OUTLET']
        UP = points['UP']
        DOWN = points['DOWN']
        CIR = points['CIR']
        U0 = points['U0']
        WALL = np.concatenate((UP, DOWN, CIR), 0)
    elif 'stepflow' in args.scenario:
        INNER = points['INNER']
        INLET = points['INLET']
        OUTLET = points['OUTLET']
        UP = points['UP']
        DOWN = points['DOWN']
        U0 = points['U0']
        WALL = np.concatenate((UP, DOWN), 0)
    else:
        sys.exit(1)

    if args.state == 'train':
        logger.info('INNER shape: ' + str(INNER.shape))
        logger.info('INLET shape: ' + str(INLET.shape))
        logger.info('OUTLET shape: ' + str(OUTLET.shape))
        logger.info('WALL shape: ' + str(WALL.shape))
        logger.info('U0 [u, v] max: ' + str(U0.max(axis=0)))
        logger.info('U0 [u, v] min: ' + str(U0.min(axis=0)))

    model = NeuralNetwork(Setting, INNER, INLET, OUTLET, WALL, U0).to(device)
    logger.info(model)

    '''COPY FILES'''
    shutil.copyfile(__file__, str(working_dir)+'/'+os.path.basename(__file__))
    working_dir.joinpath('data').mkdir(exist_ok=True)
    shutil.copyfile(data_dir, str(working_dir)+'/data/'+os.path.basename(data_dir))
    shutil.copyfile(of_data_dir, str(working_dir)+'/data/'+os.path.basename(of_data_dir))
    load_model_dir = file_dir + '/' + args.load_model_dir
    if os.path.exists(load_model_dir):
        shutil.copyfile(load_model_dir, str(working_dir)+'/0_'+os.path.basename(load_model_dir))
        logger.info("Loading PyTorch Model State from \"" + load_model_dir + "\"")
        model.load_state_dict(torch.load(load_model_dir, map_location=torch.device(device)))
        model.eval()
    else:
        logger.warning("Not exists loading Model dir \"" + load_model_dir + "\"")

    def l2error():
        pred_output_tensor = model(x_pred_tensor, y_pred_tensor, mu)
        pred_output = pred_output_tensor.cpu().detach().numpy()
        u_pred = pred_output[:,0:1]
        v_pred = pred_output[:,1:2]
        uv_mag_pred = np.sqrt(np.square(u_pred)+np.square(v_pred))
        p_pred = pred_output[:,2:3]

        u_error = u_exact-u_pred
        v_error = v_exact-v_pred
        uv_mag_error = uv_mag_exact-uv_mag_pred
        p_error = p_exact-p_pred

        # Error
        l2error_u = np.linalg.norm(u_error, 2) / np.linalg.norm(u_exact, 2)
        l2error_v = np.linalg.norm(v_error, 2) / np.linalg.norm(v_exact, 2)
        l2error_uv_mag = np.linalg.norm(uv_mag_error, 2) / np.linalg.norm(uv_mag_exact, 2)
        l2error_p = np.linalg.norm(p_error, 2) / np.linalg.norm(p_exact, 2)

        logger.info('Relative L2 error')
        logger.info('u: '+str(l2error_u))
        logger.info('v: '+str(l2error_v))
        logger.info('uv_mag: '+str(l2error_uv_mag))
        logger.info('p: '+str(l2error_p))

    if (args.state == 'train'):
        # prepare log file
        log_loss_dir = str(working_dir)+'/log_loss' + '.csv'
        with open(log_loss_dir, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            loss_header = ['epoch', 'loss', 'loss_f', 'loss_WALL', 'loss_INLET', 'loss_OUTLET']
            writer.writerow(loss_header)

        # training model
        epoch0 = 0

        for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
            epoch0 = model.train_Adam(epochs=200000,
                                        learning_rate=lr,
                                        start_time=start_time,
                                        logger=logger,
                                        epoch0=epoch0,
                                        writer_dir=log_loss_dir,
                                        write_interval=100,
                                        label_train_print=True)
            l2error()


        # for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
        #     epoch0 = model.train_batch_Adam(epochs=100000,
        #                                 learning_rate=lr,
        #                                 inner_batch_size=num_inner_batch_size,
        #                                 start_time=start_time,
        #                                 logger=logger,
        #                                 epoch0=epoch0,
        #                                 writer_dir=log_loss_dir,
        #                                 write_interval=100,
        #                                 label_train_print=True)
        #     l2error()

        logger.info("Training Done!")

    elif (args.state == 'predict'):
        if os.path.exists(load_model_dir) == False : sys.exit(1)
        import matplotlib.pyplot as plt

        def plotSolution(x_coord,y_coord,solution,file_name,title):
            plt.scatter(x_coord, y_coord, s=2.5, c=solution, cmap='jet')
            plt.title(title)
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            x_upper = np.max(x_coord)
            x_lower = np.min(x_coord)
            y_upper = np.max(y_coord)
            y_lower = np.min(y_coord)
            plt.xlim([x_lower, x_upper])
            plt.ylim([y_lower, y_upper])
            plt.gca().set_aspect(1)
            plt.colorbar(shrink=0.7)
            plt.savefig(str(working_dir)+'/'+file_name+'.png',dpi=300)
            plt.clf()
            #plt.show()
            
        l2error()
        pred_output_tensor = model(x_pred_tensor, y_pred_tensor, mu)
        pred_output = pred_output_tensor.detach().numpy()
        u_pred = pred_output[:,0:1]
        v_pred = pred_output[:,1:2]
        uv_mag_pred = np.sqrt(np.square(u_pred)+np.square(v_pred))
        p_pred = pred_output[:,2:3]

        u_error = u_exact-u_pred
        v_error = v_exact-v_pred
        uv_mag_error = uv_mag_exact-uv_mag_pred
        p_error = p_exact-p_pred

        # Plot the comparison of u, v, p
        plotSolution(x_coordinate, y_coordinate, u_pred,'u_velocity_pred','u (x-velocity component) pred')
        plotSolution(x_coordinate, y_coordinate, v_pred,'v_velocity_pred','v (y-velocity component) pred')
        plotSolution(x_coordinate, y_coordinate, uv_mag_pred,'mag_velocity_pred','magnitude velocity pred')
        plotSolution(x_coordinate, y_coordinate, p_pred,'pressure_pred','pressure pred')
        plotSolution(x_coordinate, y_coordinate, u_exact,'u_velocity_exact','u (x-velocity component) exact')
        plotSolution(x_coordinate, y_coordinate, v_exact,'v_velocity_exact','v (y-velocity component) exact')
        plotSolution(x_coordinate, y_coordinate, uv_mag_exact,'mag_velocity_exact','magnitude velocity exact')
        plotSolution(x_coordinate, y_coordinate, p_exact,'pressure_exact','pressure exact')
        plotSolution(x_coordinate, y_coordinate, u_error,'u_velocity_error','u (x-velocity component) error')
        plotSolution(x_coordinate, y_coordinate, v_error,'v_velocity_error','v (y-velocity component) error')
        plotSolution(x_coordinate, y_coordinate, uv_mag_error,'mag_velocity_error','magnitude velocity error')
        plotSolution(x_coordinate, y_coordinate, p_error,'pressure_error','pressure error')

        output_data_dir = str(working_dir)+'/output_data' + '.csv'
        with open(output_data_dir, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            loss_header = ['x_coord', 'y_coord', \
                    'u_exact', 'v_exact', 'uv_mag_exact', 'p_exact', \
                    'u_pred', 'v_pred', 'uv_mag_pred', 'p_pred', \
                    'u_error', 'v_error', 'uv_mag_error', 'p_error']
            writer.writerow(loss_header)

        output_data = np.concatenate((x_coordinate, y_coordinate, \
                    u_exact, v_exact, uv_mag_exact, p_exact, \
                    u_pred, v_pred, uv_mag_pred, p_pred, \
                    u_error, v_error, uv_mag_error, p_error), 1)
        with open(output_data_dir, 'a', encoding='UTF8', newline='') as f:
            np.savetxt(f, output_data, delimiter=',')        

        logger.info("Predict Done!")

    print('Working directory: ' + str(working_dir))
