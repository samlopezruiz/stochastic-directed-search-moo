@ECHO OFF
setlocal
set PYTHONPATH=D:/MEGA/CienciasDeLaComputacion/Tesis/StochasticDirectedSearch
FOR %%x IN (0 1 2 3 4 5 6 7 8 9) DO C:\Users\sam24\.conda\envs\machine_learning\python.exe D:/MEGA/CienciasDeLaComputacion/Tesis/TsMoo/src/timeseries/moo/cont/experiments/experiment_loop.py --model_ix %%x
endlocal

