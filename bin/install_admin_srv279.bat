REM right click, run as admin from \\srv274d1\d$\github\hhnk-fewspy\bin, not Y:\github
REM bit weird with conda activation
REM make sure site-packages doesnt have the egg-link.
del "%APPDATA%\Python\Python39\site-packages\hhnk-research-tools.egg-link" 2>null
call conda activate fewspy_env
call python -m pip install -e "\\srv279d1\d$\github\hhnk-research-tools" --no-deps

pause