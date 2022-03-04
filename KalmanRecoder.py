from abc import abstractmethod, ABCMeta
import os
from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from utils import calc_angle

class KalmanRecoder(metaclass=ABCMeta):
    """
    # Summary
    カルマンフィルタの実行と記録のためのクラス

    # Args

    """
    def __init__(self, 
                 baseDir: str, 
                 fileName: str,
                 loadRange: int,
                 use: str) -> None:
        
        self.use            = use
        self.loadRange      = loadRange
        self._calcTime_list = []
        self._estimate_list = []
        self._average_calcTime   = 0
        self._average_angleError = 0
        self.dataFile       = os.path.join(baseDir, fileName)
        self.dataFrame      = pd.read_csv(self.dataFile)

        name = fileName.split(".")[0]
        self.outputName = os.path.join(baseDir, name +\
                          f"_{self.__class__.__name__}_range{self.loadRange}_{use}.csv")
                          

    @property
    def calcTime_list(self, ):
        return self._calcTime_list

    @property
    def estimate_list(self,):
        return self._estimate_list

    @property
    def average_calcTime(self,):
        return self._average_calcTime

    @property
    def average_angleError(self,):
        return self._average_angleError
    
    @abstractmethod
    def exec_filter_with_velocity(self, initial_xy, measurements):
        """
        # Summary
        一回分のカルマンフィルタを計算する/n
        出力はList[np.ndarray] -> [x座標, y座標]

        # Args
        **kwargs: 辞書型の引数 -> {"initial_xy": initial_xy, "messurements": measurements}
        """
        pass

    @abstractmethod
    def exec_filter_with_acceleration(self, ):
        """
        # Summary
        一回分のカルマンフィルタを計算する/n
        出力はList[np.ndarray] -> [x座標, y座標]

        # Args
        **kwargs: 辞書型の引数 -> {"initial_xy": initial_xy, "messurements": measurements}
        """
        pass

    def estimate(self,):
        """
        # Summary
        カルマンフィルタを繰り返し実行し、全ての入力に対する推測値を出力する\n
        ※必ず使用するフィルタに合わせて実装するように注意

        # Args
        use: velocityかaccaccelerationを指定してください
        """
        if self.use not in ["velocity", "acceleration"]:
            raise ValueError("Please select 'velocity' or 'acceleration'")

        kalman = self.exec_filter_with_velocity if self.use == "velocity" else self.exec_filter_with_acceleration

        for i in range(len(self.dataFrame)-self.loadRange):
                initial_xy   = self.dataFrame.iloc[i, 7:9].values
                measurements = self.dataFrame.iloc[i+1:i+1+self.loadRange, 7:9].values
                self._estimate_list.append(kalman(**{"initial_xy": initial_xy,
                                                            "measurements": measurements}))

        # if self.use == "velocity":
        #     for i in range(len(self.dataFrame)-self.loadRange):
        #         initial_xy   = self.dataFrame.iloc[i, 7:9].values
        #         measurements = self.dataFrame.iloc[i+1:i+1+self.loadRange, 7:9].values
        #         self._estimate_list.append(self.exec_filter_with_velocity(**{"initial_xy": initial_xy,
        #                                                     "measurements": measurements}))

        # elif self.use == "acceleration":
        #     for i in range(len(self.dataFrame)-self.loadRange):
        #         initial_xy   = self.dataFrame.iloc[i, 7:9].values
        #         measurements = self.dataFrame.iloc[i+1:i+1+self.loadRange, 7:9].values
        #         self._estimate_list.append(self.exec_filter_with_acceleration(**{"initial_xy": initial_xy,
        #                                                     "measurements": measurements}))

    def record(self,
               result:np.ndarray,
               )->None:
        """
        # Summary
        結果が記録されたcsvファイルを入力されたファイルと同じ階層に出力する

        # Args
        result: ndarray, result[:, 0]->フィルタリング後のx座標, result[:, 1]->フィルタリング後のy座標, result[:, 2]->計算時間
        """
        outputFrame = pd.DataFrame(result)

        # 角度の計算
        lackOfRow   = len(self.dataFrame) - len(outputFrame)
        noneFrame   = pd.DataFrame([[]*2]*lackOfRow)
        empty = np.empty(lackOfRow)
        org_true = self.dataFrame.iloc[:, 7:9].values
        org_pred = np.empty_like(org_true)
        org_pred[lackOfRow:, :] = self.estimate_list
        angle_true = np.expand_dims(calc_angle(org_true), axis=1)
        angle_pred = np.expand_dims(calc_angle(org_pred), axis=1)
        print(angle_true)
        print("***************************")
        print(angle_pred)

        
        angles = np.concatenate((angle_true, angle_pred), axis=1)
        
        outputFrame = pd.concat([noneFrame, outputFrame], axis=0) \
                        .reset_index() \
                        .set_axis(["index",f"{self.__class__.__name__}_x", f"{self.__class__.__name__}_y", "calc time"], axis=1)
        angleFrame  = pd.DataFrame(angles, columns=["angle true", "angle pred"])
        outputFrame = pd.concat([outputFrame, angleFrame], axis=1)
        outputFrame = pd.concat([self.dataFrame, outputFrame.iloc[:, 1:]], axis=1)
        outputFrame.to_csv(self.outputName)

        self._average_calcTime   = np.average(self.calcTime_list)
        average_angleError = np.average(np.abs(angle_true - angle_pred), axis=1)
        self._average_angleError = np.average(average_angleError[lackOfRow+2:])

        