import glob, os
from KalmanFilters import *

if __name__ == "__main__":
    # ----------------------Test Mode-----------------------------
    testMode = False

    # ----------------------Main----------------------------------
    if not testMode:
        dataDir     =  "*/**/***"
        for baseDir in glob.glob(dataDir):
            # ----------------------- Result.csvの作成 --------------------------
            # gpsFileName       = "rawdata_GPS.csv"
            # realsenseFileName = "rawdata_realsense.csv"
            # print("[1] Result.csvの作成を開始します")
            # utils.make_result(base_file=dataFile,
            #             realsense_file=realsenseFileName,
            #             gps_file=gpsFileName)
            # print("[2] Result.csvの作成が完了しました")

            # ----------------------- 各カルマンフィルタの作成 --------------------------
            file_name = "11_Result.csv"
            loadRnage = 5
            use       = "velocity"
            use       = "acceleration"


            # **************************************************************************
            # ************************  線形カルマンフィルタ  ****************************
            # **************************************************************************


            LKF = LinearKalman(baseDir=baseDir,
                              fileName=file_name,
                              loadRange=loadRnage,
                              use=use)
            LKF.estimate()
            lk_calcTime = np.array(LKF.calcTime_list).reshape((-1, 1))
            LKF.record(np.concatenate([LKF.estimate_list, lk_calcTime], axis=1))

            # **************************************************************************
            # ************************  無香料カルマンフィルタ  **************************
            # **************************************************************************


            UK = UnscentedKalman(baseDir=baseDir,
                                 fileName=file_name,
                                 loadRange=loadRnage,
                                 use=use)
            UK.estimate()
            uk_calcTime = np.array(UK.calcTime_list).reshape((-1, 1))
            UK.record(np.concatenate([UK.estimate_list, uk_calcTime], axis=1))

            # **************************************************************************
            # ************************  パーティクルフィルタ  ****************************
            # **************************************************************************


            PF = ParticleFilter(baseDir=baseDir,
                                 fileName=file_name,
                                 loadRange=loadRnage,
                                 use=use)
            PF.estimate()
            pf_calcTime = np.array(PF.calcTime_list).reshape((-1, 1))
            PF.record(np.concatenate([PF.estimate_list, pf_calcTime], axis=1))

            # **************************************************************************
            # ************************  拡張カルマンフィルタ  ****************************
            # **************************************************************************


            EKF = ExtendedKalman(baseDir=baseDir,
                                 fileName=file_name,
                                 loadRange=loadRnage,
                                 use=use)
            EKF.estimate()
            ekf_calcTime = np.array(EKF.calcTime_list).reshape((-1, 1))
            EKF.record(np.concatenate([EKF.estimate_list, ekf_calcTime], axis=1))

            timeFrame = pd.DataFrame([[LKF.average_calcTime, LKF.average_angleError],
                                      [UK.average_calcTime , UK.average_angleError ],
                                      [PF.average_calcTime , PF.average_angleError ],
                                      [EKF.average_calcTime, EKF.average_angleError]], 
                             columns=["average calc time", "average angle error"],
                             index=[LKF.__class__.__name__,
                                      UK.__class__.__name__,
                                      PF.__class__.__name__,
                                      EKF.__class__.__name__])
            timeFrame.to_csv(os.path.join(baseDir, f"average_calc_time_{use}.csv"))





            
    else:
        dataDir     =  "*/**/***"
        target      = glob.glob(dataDir)[0]
        outputName  = os.path.join(target, "result.csv")
        print("***************************************************")
        print("[*] 出力ファイルの出力先が正しいことを確認してください")
        print("[*] 出力先に問題がなければmain.py中にある'testMode'をFalseに変更してください")
        print(f"[1] 出力先 : {outputName}")
        print("***************************************************")