@ECHO OFF
setlocal
set PYTHONPATH=D:/MEGA/CienciasDeLaComputacion/Tesis/TsMoo/
FOR %%x IN (11 12 13 15) DO C:\Users\sam24\.conda\envs\machine_learning\python.exe D:/MEGA/CienciasDeLaComputacion/Tesis/TsMoo/src/timeseries/moo/cont/experiments/experiment_loop.py --model_ix %%x
endlocal

