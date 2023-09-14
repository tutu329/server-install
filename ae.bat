@echo off
set filename="c:\apache24\logs\error.log"
powershell -command "get-content %filename% -ReadCount 0 -Tail 5 -Wait"