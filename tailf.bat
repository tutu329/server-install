@echo off
set filename=%1
powershell -command "get-content %filename% -ReadCount 0 -Tail 5 -Wait"