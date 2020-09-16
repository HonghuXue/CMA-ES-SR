'''
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.



tau keeps the same trajectory shape, only with the effect of elogating/shortening evenly the whole trajectory
rollout(timesteps) 
cs.runtime makes sure that the rbf is evenly distributed in Z space ,   e.g. cs.run_time =1 , tau = 1, timesteps = 500, num_rbf = 50   v.s. cs.run_time =5. , tau = 1, timesteps = 500, num_rbf = 50
'''
import math as ms
from dmp import DMPs
import matplotlib.pyplot as plt
import numpy as np


class DMPs_discrete(DMPs):
    """An implementation of discrete DMPs"""

    def __init__(self, **kwargs):
        """
        """

        # call super class constructor
        super(DMPs_discrete, self).__init__(pattern='discrete', **kwargs)

        self.gen_centers()

        # set variance of Gaussian basis functions
        # trial and error to find this spacing
        self.h = np.ones(self.n_bfs) * self.n_bfs**1.5 / self.c / self.cs.ax

        self.check_offset()

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""

        '''x_track = self.cs.discrete_rollout()
        t = np.arange(len(x_track))*self.dt
        # choose the points in time we'd like centers to be at
        c_des = np.linspace(0, self.cs.run_time, self.n_bfs)
        self.c = np.zeros(len(c_des))
        for ii, point in enumerate(c_des):
            diff = abs(t - point)
            self.c[ii] = x_track[np.where(diff == min(diff))[0][0]]'''

        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)

        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        return x * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """

        return np.copy(y_des[:, -1])

    def gen_psi(self, x):
        """Generates the activity of the basis functions for a given
        canonical system rollout.

        x float, array: the canonical system state or path
        """

        if isinstance(x, np.ndarray):
            x = x[:, None]
        return np.exp(-self.h * (x - self.c)**2)

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # calculate x and psi
        x_track = self.cs.rollout()
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term
            k = (self.goal[d] - self.y0[d])
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track**2 * psi_track[:, b])
                self.w[d, b] = numer / (k * denom)
        self.w = np.nan_to_num(self.w)

# ==============================
# Test code
# ==============================

def alpha_beta_rescaling(alpha, conf):
    slope = 0.1
    if isinstance(alpha, np.ndarray):
        alpha_rescaled= np.ones(len(alpha))
        for i in range(len(alpha)):
            if alpha[i] >= (conf['alpha'][1] - conf['alpha'][0])/(2*slope):
                alpha_rescaled[i] = conf['alpha'][1]
            elif alpha[i] <= (-conf['alpha'][1] + conf['alpha'][0])/(2*slope):
                alpha_rescaled[i] = conf['alpha'][0]
            else:
                alpha_rescaled[i] =  slope * alpha[i] + (conf['alpha'][1] + conf['alpha'][0])/2
    else:
        if alpha >= (conf['alpha'][1] - conf['alpha'][0])/(2*slope):
            alpha_rescaled = conf['alpha'][1]
        elif alpha <= (-conf['alpha'][1] + conf['alpha'][0])/(2*slope):
            alpha_rescaled = conf['alpha'][0]
        else:
            alpha_rescaled =  slope * alpha + (conf['alpha'][1] + conf['alpha'][0])/2
    return alpha_rescaled   



if __name__ == "__main__":
#2.5 * np.random.randn(2, 4) + 3

    np.random.seed(53)    
    # test normal run
     # weight+ alpha  as parameters, goal as known. total time (timesteps)/y_0 should be flexible 
    xbest = np.array([-2.12160394e+00, -1.03379942e+02, -1.07949795e+01, -6.22359611e+00,
                    8.13203111e+01,  2.20517536e+01, -7.22776346e+00,  1.86511255e+01,
                   -2.14643971e+01, -2.18886974e+01,  1.42319997e+01,  2.09153086e+01,
                    4.56807722e-02, -3.36967077e+01, -8.36501022e+01, -1.92644430e+01,
                    2.53829304e+01, -3.23140033e+01,  2.63074237e+01,  1.27645526e+01,
                    1.18781721e+01,  2.19705093e+01, -2.17995724e+01, -7.25830921e+01,
                   -7.14705082e+00,  4.66843099e+01, -1.07791364e+02,  1.49927866e+01,
                   -3.08104463e+01, -2.85958190e+01, -4.60744794e+01, -4.32698531e+01,
                    4.55460378e+01,  1.04843272e+01, -2.65259041e+01, -5.04600813e+01,
                    2.58020795e+01, -5.21275676e+01,  8.62869243e+01,  1.86767501e+01,
                   -2.27426856e+01,  2.78655797e+01, -5.15841635e+01, -3.51362704e+01,
                    4.68584939e+00,  5.17552693e+01, -3.75468948e+01, -2.18857208e+01,
                    8.07777579e+00, -8.32548715e+01,  3.23795108e+01,  6.35936132e+01,
                    7.46312761e+00,  1.91460516e+01, -3.50151933e+00, -6.22880854e+00,
                   -4.70619080e+01, -2.50270060e+01,  4.46284919e+01, -2.35634182e+01,
                   -2.14580330e+01,  7.94132396e+01, -1.45894028e+01,  3.23139808e+01,
                    6.16964142e+00,  4.36111056e-01, -9.17555292e+00, -4.08896025e+00,
                   -5.60072959e+01, -1.84944298e+01,  7.26386380e+00, -3.74726216e+01,
                    3.46882461e+01,  3.33036381e+01,  8.05888162e+00,  3.02015268e+01,
                    2.01587939e+00, -6.34286954e+01, -7.19483435e+01,  4.45529059e+00,
                    2.64171401e+01,  4.38503852e-01, -2.13531497e+01, -8.34263430e+01]).reshape(7,-1)
    conf = {'initial_joint_angles': np.array([0., 0., 0., -1./18.*ms.pi, 0., ms.pi* 10/18, 2./9.*ms.pi]),
            'angular_accel_constraint': np.array([15, 7.5, 10, 12.5, 15, 20, 20])/ms.pi * 180,
            'alpha':[1., 20.], 'beta':[1.,20.]}  
    
    n_dmps = 1 
    n_bfs = 10
    slow_down_multitude = 1
    tau = 0.1 * 1./slow_down_multitude
    save = True
    dmp = DMPs_discrete(dt=.01, n_dmps=n_dmps, n_bfs=n_bfs, w=np.random.normal(0, 20, size = (n_dmps, n_bfs)), goal=4, ay=40 * np.ones(n_dmps), y0=6)
    y_track, dy_track, ddy_track = dmp.rollout(timesteps = round(500/tau), tau = tau)  
    plt.figure(1, figsize=(6, 3))
    plt.plot(y_track, lw=2)
    plt.title('Angle profile')
    plt.xlabel('time (ms)')
    plt.ylabel('Angle [deg]')
    plt.legend(['goal', 'system state'], loc='lower right')
    plt.tight_layout()
    
    
#    ay=alpha_beta_rescaling(xbest[-1][-1]*np.ones(n_dmps), conf)
    #------------------------------------------------------------Plot each joint angle
#    for i in range(7):
#        dmp = DMPs_discrete(dt=.01, n_dmps=1, n_bfs=n_bfs, w=np.expand_dims(xbest[i][:-2], axis=0), goal=xbest[i][-2]+ conf['initial_joint_angles'][i]*180 /ms.pi, ay=alpha_beta_rescaling(xbest[i][-1]*np.ones(n_dmps), conf), by=alpha_beta_rescaling(xbest[i][-1]*np.ones(n_dmps), conf)/4., y0=conf['initial_joint_angles'][i]*180 /ms.pi)# weight as parameters, goal as known
#        y_track, dy_track, ddy_track = dmp.rollout(timesteps = round(500/tau), tau = tau)   
#    
#        plt.figure(i+1, figsize=(6, 3))
##        plt.plot(np.ones(len(ddy_track)) * conf['angular_accel_constraint'][i], 'r--', lw=2)
##        plt.plot(-np.ones(len(ddy_track)) * conf['angular_accel_constraint'][i], 'r--', lw=2)
#        plt.plot(y_track, lw=2)
#        plt.title('Angle profile')
#        plt.xlabel('time (ms)')
#        plt.ylabel('Angle [deg]')
#        plt.legend(['goal', 'system state'], loc='lower right')
#        plt.tight_layout()
#
#
#    if save == True:
#        dmp = DMPs_discrete(dt=.01, n_dmps=7, n_bfs=n_bfs, w=xbest[:,:-2], goal=xbest[:,-2]+ conf['initial_joint_angles']*180 /ms.pi, ay=alpha_beta_rescaling(xbest[:,-1]*np.ones(n_dmps), conf), by=alpha_beta_rescaling(xbest[:,-1]*np.ones(n_dmps), conf)/4., y0=conf['initial_joint_angles']*180 /ms.pi)# weight as parameters, goal as known
#        y_track, dy_track, ddy_track = dmp.rollout(timesteps = round(500/tau), tau = tau)     
#        np.savetxt('Joint_angle.txt', y_track, delimiter = ';' , fmt = '%10.5f')
    #------------------------------------------------------------Plot each joint angle
#    plt.figure(2, figsize=(6, 3))
#    plt.plot(np.ones(len(y_track))*dmp.goal, 'r--', lw=2)
#    plt.plot(dy_track, lw=2)
#    plt.title('Angular Velocity Profile')
#    plt.xlabel('time (ms)')
#    plt.ylabel('\u03C9  [deg/s]')
#    plt.legend(['goal', 'system state'], loc='lower right')
#    plt.tight_layout()

#    # test imitation of path run
#    plt.figure(2, figsize=(6, 4))
#    n_bfs = [10]
#
#    # a straight line to target
#    path1 = np.sin(np.arange(0, 1, .01)*5)
#    # a strange path to target
#    path2 = np.zeros(path1.shape)
#    path2[int(len(path2) / 2.):] = .5
#
#    for ii, bfs in enumerate(n_bfs):
#        dmp = DMPs_discrete(n_dmps=2, n_bfs=bfs)
#
#        dmp.imitate_path(y_des=np.array([path1, path2]))
#        # change the scale of the movement
#        dmp.goal[0] = 3
#        dmp.goal[1] = 2
#
#        y_track, dy_track, ddy_track = dmp.rollout()
#
#        plt.figure(2)
#        plt.subplot(211)
#        plt.plot(y_track[:, 0], lw=2)
#        plt.subplot(212)
#        plt.plot(y_track[:, 1], lw=2)
#
#    plt.subplot(211)
#    a = plt.plot(path1 / path1[-1] * dmp.goal[0], 'r--', lw=2)
#    plt.title('DMP imitate path')
#    plt.xlabel('time (ms)')
#    plt.ylabel('system trajectory')
#    plt.legend([a[0]], ['desired path'], loc='lower right')
#    plt.subplot(212)
#    b = plt.plot(path2 / path2[-1] * dmp.goal[1], 'r--', lw=2)
#    plt.title('DMP imitate path')
#    plt.xlabel('time (ms)')
#    plt.ylabel('system trajectory')
#    plt.legend(['%i BFs' % i for i in n_bfs], loc='lower right')
#
#    plt.tight_layout()
#    plt.show()

    plt.figure()
    psi_track = dmp.gen_psi(dmp.cs.rollout())
    plt.plot(psi_track)
    