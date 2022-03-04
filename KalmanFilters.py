from matplotlib.pyplot import axis
from KalmanRecoder import KalmanRecoder
from udacity_kalman_filter import  matrix
import pandas as pd
import numpy as np
import utils, os, scipy
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from filterpy.monte_carlo import systematic_resample
from numpy.random import uniform, randn, random


class LinearKalman(KalmanRecoder):
    def __init__(self, 
                 baseDir: str, 
                 fileName: str,
                 loadRange: int,
                 use: str):
        super().__init__(baseDir, fileName, loadRange, use)
        
    @utils.measure_time
    def exec_filter_with_velocity(self, initial_xy, measurements)->np.ndarray:
        """
        # Summary
        線形カルマンフィルタを実行する
        """

        dt = 0.1

        # x:初期状態（位置と速度)
        x = matrix([[initial_xy[0]], [initial_xy[1]], [0.], [0.]])
        # u:外部動作
        u = matrix([[0.], [0.], [0.], [0.]]) 

        # P:初期不確かさ
        P = matrix([[5000., 0., 0., 0.],
                    [0., 5000., 0., 0.],
                    [0., 0., 25., 0.],
                    [0., 0., 0., 25.]])
        # F:次の状態遷移関数
        F = matrix([[1., 0., dt, 0.],
                    [0., 1., 0., dt],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
        # H:測定関数
        H = matrix([[1., 0., 0., 0.],
                    [0., 1., 0., 0.]]) 
        # R:測定の不確かさ
        R = matrix([[0.1, 0.],
                    [0., 0.1]]) 
        # I:4D単位行列
        I = matrix([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]]) 

        for n in range(len(measurements)):
            
            
            # measurement update
            Z = matrix([measurements[n]])
            y = Z.transpose() - (H * x)
            S = H * P * H.transpose() + R
            K = P * H.transpose() * S.inverse()
            x = x + (K * y)
            P = (I - (K * H)) * P
        
            # prediction
            x = (F * x) + u
            P = F * P * F.transpose()

        output = x.value[:2]
        return np.array([output[i][0] for i in range(len(output))])

    @utils.measure_time
    def exec_filter_with_acceleration(self, initial_xy, measurements)->np.ndarray:
        """
        # Summary
        線形カルマンフィルタを実行する
        """

        dt = 0.1

        # x:初期状態（位置と速度)
        x = matrix([[int(initial_xy[0])],[0], [0], [int(initial_xy[1])], [0], [0]])
        # u:外部動作
        u = matrix([[0.], [0.], [0.], [0.], [0.], [0.]]) 

        # P:初期不確かさ
        P = matrix([[5000., 0., 0., 0., 0, 0],
                    [0., 25., 0., 0., 0, 0],
                    [0., 0., 5., 0., 0, 0],
                    [0., 0., 0., 5000., 0, 0],
                    [0., 0., 0., 0., 25, 0],
                    [0., 0., 0., 0., 0, 5]])
        # F:次の状態遷移関数
        F = matrix([[1, dt, 0.5*dt**2,      0, 0, 0],
                      [0,  1,        dt,      0, 0, 0],
                      [0,  0,         1,      0, 0, 0],
                      [0,  0, 0,      1, dt, 0.5*dt**2],
                      [0,  0, 0,      0,        1, dt],
                      [0,  0, 0,      0,        0,  1]])
        # H:測定関数
        H = matrix([[1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0.]]) 
        # R:測定の不確かさ
        R = matrix([[0.1, 0.],
                    [0., 0.1]]) 
        # I:4D単位行列
        I = matrix(np.eye(6)) 

        for n in range(len(measurements)):
            
            
            # measurement update
            Z = matrix([measurements[n]])
            y = Z.transpose() - (H * x)
            S = H * P * H.transpose() + R
            K = P * H.transpose() * S.inverse()
            x = x + (K * y)
            P = (I - (K * H)) * P
        
            # prediction
            x = (F * x) + u
            P = F * P * F.transpose()

        output =[x.value[0], x.value[3]]
        return np.array([output[i][0] for i in range(len(output))])

class UnscentedKalman(KalmanRecoder):
    def __init__(self, 
                 baseDir: str, 
                 fileName: str,
                 loadRange: int,
                 use: str) -> None:
        super().__init__(baseDir, fileName, loadRange, use)

        self.dt  = 1
        self.std = 0.1

    
    @utils.measure_time
    def exec_filter_with_acceleration(self, **kwargs)->np.ndarray:
        """
        # Summary
        無香料カルマンフィルタを実行する
        """
        initial_xy   = kwargs["initial_xy"]
        measurements = kwargs["measurements"]
        

        sigmas = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=1.)
        ukf = UKF(dim_x=6, dim_z=2, fx=self._f_cv_acc,
                hx=self._h_cv_acc, dt=self.dt, points=sigmas)
        ukf.x = np.array([initial_xy[0], 0., 0., initial_xy[1], 0., 0.])
        ukf.R = np.diag([self.std, self.std])
        ukf.Q[:3, :3] = Q_discrete_white_noise(3, dt=self.dt, var=0.02)
        ukf.Q[3:, 3:] = Q_discrete_white_noise(3, dt=self.dt, var=0.02)

        for measurement in measurements:
            ukf.predict()
            ukf.update(measurement)

        return ukf.x[[0, 2]]

    @utils.measure_time
    def exec_filter_with_velocity(self, **kwargs)->np.ndarray:
        """
        # Summary
        無香料カルマンフィルタを実行する
        """
        initial_xy   = kwargs["initial_xy"]
        measurements = kwargs["measurements"]
        

        sigmas = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=1.)
        ukf = UKF(dim_x=4, dim_z=2, fx=self._f_cv_vel,
                hx=self._h_cv_vel, dt=self.dt, points=sigmas)
        ukf.x = np.array([initial_xy[0], 0., initial_xy[1], 0.])
        ukf.R = np.diag([self.std, self.std])
        ukf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=self.dt, var=0.02)
        ukf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=self.dt, var=0.02)

        for measurement in measurements:
            ukf.predict()
            ukf.update(measurement)

        return ukf.x[[0, 2]]

    def _f_cv_vel(self, x, dt):
        F = np.array([[1, dt, 0,  0],
                      [0,  1, 0,  0],
                      [0,  0, 1, dt],
                      [0,  0, 0,  1]])
        return F @ x

    def _f_cv_acc(self, x, dt):
        F = np.array([[1, dt, 0.5*dt**2,      0, 0, 0],
                      [0,  1,        dt,      0, 0, 0],
                      [0,  0,         1,      0, 0, 0],
                      [0,  0, 0,      1, dt, 0.5*dt**2],
                      [0,  0, 0,      0,        1, dt],
                      [0,  0, 0,      0,        0,  1]])
        return F @ x

    def _h_cv_acc(self, x):
        return x[[0, 3]]

    def _h_cv_vel(self, x):
        return x[[0, 2]]

    

class ExtendedKalman(KalmanRecoder):
    """
    # Summary
    拡張カルマンフィルタを実行する

    # Property
    self.dt : サンプリング周期(例：100Hzでデータを取得したなら1/100)
    self.std: 測定誤差[m]
    """
    def __init__(self, 
                 baseDir: str, 
                 fileName: str,
                 loadRange: int,
                 use: str) -> None:
        super().__init__(baseDir, fileName, loadRange, use)

        self.dt  = 1
        self.std = 0.1



    @utils.measure_time
    def exec_filter_with_velocity(self, **kwargs):
        initial_xy   = kwargs["initial_xy"]
        measurements = kwargs["measurements"]

        ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
        ekf.x = np.array([initial_xy[0], 0., initial_xy[1], 0.])
        ekf.F = np.array([[1, self.dt, 0,       0],
                          [0,       1, 0,       0],
                          [0,       0, 1, self.dt],
                          [0,       0, 0,       1]])
        ekf.R = np.diag([self.std, self.std])
        ekf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=self.dt, var=0.02)
        ekf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=self.dt, var=0.02)
        ekf.P *= 50

        for measurement in measurements:
            ekf.predict()
            ekf.update(measurement, self._HJacobian_vel ,self._H_vel)

        return ekf.x[[0, 2]]

    def _H_vel(self, x):
        return x[[0, 2]]

    def _HJacobian_vel(self, x):
        return np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]])
                         
    @utils.measure_time
    def exec_filter_with_acceleration(self, **kwargs):
        initial_xy   = kwargs["initial_xy"]
        measurements = kwargs["measurements"]

        ekf = ExtendedKalmanFilter(dim_x=6, dim_z=2)
        ekf.x = np.array([initial_xy[0], 0., 0, initial_xy[1], 0., 0])
        ekf.F = np.array([[1, self.dt, 0.5*self.dt**2,      0, 0, 0],
                      [0,  1,        self.dt,      0, 0, 0],
                      [0,  0,         1,      0, 0, 0],
                      [0,  0, 0,      1, self.dt, 0.5*self.dt**2],
                      [0,  0, 0,      0,        1, self.dt],
                      [0,  0, 0,      0,        0,  1]])
        ekf.R = np.diag([self.std, self.std])
        ekf.Q[0:3, 0:3] = Q_discrete_white_noise(3, dt=self.dt, var=0.02)
        ekf.Q[3:6, 3:6] = Q_discrete_white_noise(3, dt=self.dt, var=0.02)
        ekf.P *= 50

        for measurement in measurements:
            ekf.predict()
            ekf.update(measurement, self._HJacobian_acc ,self._H_acc)

        return ekf.x[[0, 3]]

    def _H_acc(self, x):
        return x[[0, 3]]

    def _HJacobian_acc(self, x):
        return np.array([[1, 0, 0, 0, 0 ,0],
                         [0, 0, 1, 0, 0, 0]])


class ParticleFilter(KalmanRecoder):
    """
    # Summary
    モンテカルロ法（ランダムに生成された有限個のサンプルを使って計算する手法）を用いたフィルタ
    ## Particle Filterの手順\n
    1. 粒子の生成
    2. 予測
    3. 更新
    4. 再サンプリング
    5. 推定値の計算

    """
    def __init__(self, 
                 baseDir: str, 
                 fileName: str,
                 loadRange: int,
                 use: str) -> None:
        super().__init__(baseDir, fileName, loadRange, use)

        self.dt = 1.
        self.N = 100
        self.v_upd = 0
        self.v_pred = 0
        self.alpha = 0.7
        self.mean = np.array([0, 0])
        self.var = np.array([0.01, 0.01])


    @utils.measure_time
    def exec_filter_with_velocity(self, **kwargs):
        initial_xy   = np.array(kwargs["initial_xy"])
        measurements = np.array(kwargs["measurements"])
        measurements = np.concatenate([np.expand_dims(initial_xy, axis=0), measurements], axis=0)

        self.create_gaussian_particles(mean=initial_xy, std=[0.1, 0.1])
        self.weights = np.reshape(np.ones(self.N) / self.N, (-1, 1))

        for i in range(measurements.shape[0] - 1):
            self.predict(measurements[i], measurements[i+1], std=0.1)
            self.update(measurements[i], measurements[i+1], R=0.1)
            self.resample()
            self.estimate_()

        return self.mean


    def create_uniform_particles(self, x_range, y_range,):
        """
        # Summary
        1. パーティクルの生成
        """
        self.particles = np.empty((self.N, 3))
        self.particles[:, 0] = uniform(x_range[0], x_range[1], size=self.N)
        self.particles[:, 1] = uniform(y_range[0], y_range[1], size=self.N)

    def create_gaussian_particles(self, mean, std,):
        """
        # Summary
        1. パーティクルの生成
        """
        self.particles = np.empty((self.N, 3))
        self.particles[:, 0] = mean[0] + (randn(self.N) * std[0])
        self.particles[:, 1] = mean[1] + (randn(self.N) * std[1])

    def predict(self, initial_xy, u, std,):
        """
        # Summary
        2. 予測
        速度と前回の位置から
        """
        # 位置の更新
        v = u - initial_xy
        prediction = initial_xy + v * self.dt
        dist = np.reshape(prediction, (-1, 2)) + np.reshape(randn(self.N*2) * std, (-1, 2))
        self.particles[:, 0] += dist[:, 0]
        self.particles[:, 1] += dist[:, 1]

    def update(self, initial_xy, z, R,):
        """
        # Summary
        3. パーティクルの位置の更新
        ※観測値と予測値の平均値を用いて重みづけを行う
        """
        v = z - initial_xy
        prediction = initial_xy + v * self.dt
        predict = np.reshape((self.alpha * z + (1 - self.alpha) * prediction) / 2, (-1,2))
        distance = np.reshape(np.linalg.norm(self.particles[:, 0:2] - predict, axis=1), (-1, 1))
        distance_obs = np.reshape(np.linalg.norm(self.particles[:, 0:2] - z, axis=1), (-1, 1))
        self.weights = np.reshape(self.weights, (-1,1)) * np.reshape(scipy.stats.norm(distance, R).pdf(distance_obs), (-1, 1))

        self.weights += 1.e-300      # 打ち切り誤差で生じる 0 を防ぐ。
        self.weights /= sum(self.weights) # 正規化する。

    def resample(self,):
        """
        # Summary
        4. 再サンプリング
        """
        if self.neff() < self.N/2:
            indexes = systematic_resample(self.weights)
            self.particles[:] = self.particles[indexes]
            self.weights.resize(len(self.particles))
            self.weights.fill(1.0 / len(self.weights))

    def neff(self, ):
        """
        # Summary
        実行サンプルサイズの計算
        """
        return 1. / np.sum(np.square(self.weights))

    def estimate_(self,):
        """
        # Summary
        5. 推定
        """
        w = self.weights.reshape(self.N, 1)
        pos = self.particles[:, 0:2]
        self.mean = np.sum(pos * w, axis=0)
        self.var  = np.sum((pos - self.mean)**2 * w, axis=0)

    @utils.measure_time
    def exec_filter_with_acceleration(self, **kwargs):
        initial_xy   = np.array(kwargs["initial_xy"])
        measurements = np.array(kwargs["measurements"])
        measurements = np.concatenate([np.expand_dims(initial_xy, axis=0), measurements], axis=0)

        self.create_gaussian_particles(mean=initial_xy, std=[0.1, 0.1])
        self.weights = np.reshape(np.ones(self.N) / self.N, (-1, 1))

        for i in range(measurements.shape[0] - 1):
            self.predict_acc(measurements[i], measurements[i+1], std=0.1)
            self.update_acc(measurements[i], measurements[i+1], R=0.1)
            self.resample()
            self.estimate_()

        return self.mean

    def predict_acc(self, initial_xy, u, std,):
        """
        # Summary
        2. 予測
        速度と前回の位置から
        """
        # 位置の更新
        v = u - initial_xy
        a = v - self.v_pred
        self.v_pred = v
        prediction = initial_xy + v * self.dt + 0.5 * a * self.dt**2
        dist = np.reshape(prediction, (-1, 2)) + np.reshape(randn(self.N*2) * std, (-1, 2))
        self.particles[:, 0] += dist[:, 0]
        self.particles[:, 1] += dist[:, 1]

    def update_acc(self, initial_xy, z, R,):
        """
        # Summary
        3. パーティクルの位置の更新
        ※観測値と予測値の平均値を用いて重みづけを行う
        """
        v = z - initial_xy
        a = v - self.v_upd
        self.v_upd = v
        prediction = initial_xy + v * self.dt + 0.5 * a * self.dt**2
        predict = np.reshape((self.alpha * z + (1 - self.alpha ) * prediction) / 2, (-1,2))
        distance = np.reshape(np.linalg.norm(self.particles[:, 0:2] - predict, axis=1), (-1, 1))
        distance_obs = np.reshape(np.linalg.norm(self.particles[:, 0:2] - z, axis=1), (-1, 1))
        self.weights = np.reshape(self.weights, (-1,1)) * np.reshape(scipy.stats.norm(distance, R).pdf(distance_obs), (-1, 1))

        self.weights += 1.e-300      # 打ち切り誤差で生じる 0 を防ぐ。
        self.weights /= sum(self.weights) # 正規化する。
    