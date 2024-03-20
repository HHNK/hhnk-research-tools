REM run as admin from \\corp.hhnk.nl\data\Hydrologen_data\Data\github\wvangerwen\hhnk-research-tools\bin, not Y:\github
REM make sure site-packages doesnt have the egg-link.
del "%APPDATA%\Python\Python39\site-packages\hhnk-research-tools.egg-link" 2>null
call conda activate fewspy_env
call python -m pip install "\\corp.hhnk.nl\data\Hydrologen_data\Data\github\wvangerwen\hhnk-research-tools" --no-deps

pause