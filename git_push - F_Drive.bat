@echo off

F:

cd "F:\05 GIT\01_masterRepo"
git pull
git add .
git commit -m "01_masterRepo_auto_push"
git push


cd "F:\05 GIT\14_Proj_ML_99Acres"
git pull
git add .
git commit -m "14_Proj_ML_99Acres_auto_push"
git push


cd "F:\05 GIT\13_Misc_projects"
git pull
git add .
git commit -m "13_Misc_projects_auto_push"
git push


cd "F:\05 GIT\12_GenAI_projects"
git pull
git add .
git commit -m "12_GenAI_projects_auto_push"
git push


cd "F:\05 GIT\11_NLP_projects"
git pull
git add .
git commit -m "11_NLP_projects_auto_push"
git push


cd "F:\05 GIT\10_DL_projects"
git pull
git add .
git commit -m "10_DL_projects_auto_push"
git push


cd "F:\05 GIT\09_ML_projects"
git pull
git add .
git commit -m "09_ML_projects_auto_push"
git push

pause

exit