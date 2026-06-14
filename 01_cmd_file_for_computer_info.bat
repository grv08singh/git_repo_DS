:: cmd for computer info in an html file:
(echo ^<html^>^<body^>^<pre^> && systeminfo && echo ^</pre^>^</body^>^<html^>) > "%USERPROFILE%\Desktop\pc_details_cmd.html"

:: cmd for battery report in an html file:
powercfg /batteryreport /output "%userprofile%\Desktop\battery-report.html"
