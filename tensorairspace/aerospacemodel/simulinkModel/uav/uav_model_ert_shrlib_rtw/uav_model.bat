set MATLAB=C:\Program Files\Polyspace\R2015B

cd .

if "%1"=="" (C:\PROGRA~1\POLYSP~1\R2015B\bin\win64\gmake -f uav_model.mk all) else (C:\PROGRA~1\POLYSP~1\R2015B\bin\win64\gmake -f uav_model.mk %1)
@if errorlevel 1 goto error_exit

exit /B 0

:error_exit
echo The make command returned an error of %errorlevel%
An_error_occurred_during_the_call_to_make
