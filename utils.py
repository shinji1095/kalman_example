import os
import pandas as pd
import numpy as np
from typing import List
from udacity_kalman_filter import matrix
from time import time

def measure_time(func):
    """
    # Summary
    カルマンフィルタの実行時間を計測
    """
    def wrapper(self, **kwargs):
        start = time()
        result = func(self, **kwargs)
        end = time()
        self.calcTime_list.append(end - start)
        return result

    return wrapper

def calc_angle(xy):
    delta = 1e-6

    previous = np.zeros_like(xy)
    previous[1:] = xy[:-1]
    residual = xy - previous


    angle = np.abs(np.arctan(residual[:,1] / (residual[:,0] + delta)) / np.pi * 180)
    angle = np.where(residual[:,0] >= 0, angle, 180 - angle)
    angle = np.where(residual[:,1] >= 0, angle, -angle)
    return angle






def calc_matrix(initial_xy, measurements)->List[np.ndarray]:
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
    return [output[i][0] for i in range(len(output))]


def make_result(base_dir, realsense_file, gps_file)->None:
    # -----------------ファイル名の定義-----------------------------
    outputFileName    = os.path.join(base_dir, "3-1_realsense_result.csv")
    realsenseFileName = os.path.join(base_dir, realsense_file)
    gpsFileName       = os.path.join(base_dir, gps_file)

    # -----------------DataFrameの定義-----------------------------
    realsense_frame = pd.read_csv(realsenseFileName)
    gps_frame       = pd.read_csv(gpsFileName)
    gps_time        = gps_frame.iloc[:,0]

    # -----------------ファイル名の定義-----------------------------
    xx, _ = np.meshgrid(realsense_frame["Time"].values, gps_time.values)
    meshgrid_diff = pd.DataFrame(np.abs(gps_time.values.reshape((-1, 1)) - xx))
    min_idx = meshgrid_diff.idxmin(axis=1)

    # -----------------結果の出力-----------------------------
    yaw_list = []
    for idx in min_idx.values:
        value = realsense_frame["yaw"][idx]
        yaw_list.append(value)
    result_frame = pd.DataFrame(yaw_list).join(gps_time)
    result_frame.set_axis(["yaw", "Time"], axis=1).reindex(columns=["Time", "yaw"]).to_csv(outputFileName)



def task_1(base_dir, file_name, load_range):
    """
    # Summary
    1. load_range分のデータを読み込む
    2. calc_matrixで計算した値の前から2つ分のデータを取得
    3. resultにデータを保存
    4. 1行読みこむデータをずらして1.に戻る
    5. すべての計算が終わったらfile_nameのcsvファイルに結果を保存

    # args
    base_dir  : 読み込みたいファイルがあるディレクトリ
    file_name : 読み込みたいファイル名
    slice_num : いくつのデータを使用するか

    # Memo
    出力データ数は(入力データの行数) - (load_range -1)（例）入力データ数 = 47, load_range = 3のとき, 出力データ数 = 45

    """
    # ********************************************************
    # ******************* ファイル名の定義 ********************
    # ********************************************************
    name = file_name.split(".")[0]
    outputName = os.path.join(base_dir, name + f"_kalman_{load_range}.csv") 
    dataFile   = os.path.join(base_dir, file_name)
    dataFrame  = pd.read_csv(dataFile)

    # ********************************************************
    # *******************     Main     ***********************
    # ********************************************************
    result = []
    load_range -= 1

    for i in range(len(dataFrame)-load_range):
        initial_xy   = dataFrame.iloc[i, 7:9].values
        measurements = dataFrame.iloc[i+1:i+1+load_range, 7:9].values
        result.append(calc_matrix(initial_xy, measurements))
    
    # ********************************************************
    # ******************* 結果の出力 **************************
    # ********************************************************
    outputFrame = pd.DataFrame(result)
    noneFrame   = pd.DataFrame([[]*2]*load_range)
    outputFrame = pd.concat([noneFrame, outputFrame], axis=0) \
                    .reset_index() \
                    .set_axis(["index",f"kalman_{load_range+1}_x", f"kalman_{load_range+1}_y"], axis=1)
    outputFrame = pd.concat([dataFrame, outputFrame.iloc[:, 1:]], axis=1)
    outputFrame.to_csv(outputName)
    