set MATLAB=D:\MATLAB2015

cd .

if "%1"=="" (D:\MATLAB2015\bin\win64\gmake -f elv_model.mk all) else (D:\MATLAB2015\bin\win64\gmake -f elv_model.mk %1)
@if errorlevel 1 goto error_exit

exit /B 0

:error_exit
echo The make command returned an error of %errorlevel%
An_error_occurred_during_the_call_to_make
