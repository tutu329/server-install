@echo off
set filename="c:\apache24\python_user.log"
powershell -command "get-content %filename% -ReadCount 0 -Tail 5 -Wait"