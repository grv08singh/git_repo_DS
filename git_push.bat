@echo off

D:

cd "D:\05 GIT\15_AI_Travel_Agent"
git pull
git add .
git commit -m "15_AI_Travel_Agent_auto_push"
git push

cd "D:\05 GIT\01_masterRepo"
git pull
git add .
git commit -m "01_masterRepo_auto_push"
git push


cd "D:\05 GIT\02_Excel_projects"
git pull
git add .
git commit -m "02_Excel_projects_auto_push"
git push


cd "D:\05 GIT\03_SQL_projects"
git pull
git add .
git commit -m "03_SQL_projects_auto_push"
git push


cd "D:\05 GIT\04_PowerBI_projects"
git pull
git add .
git commit -m "04_PowerBI_projects_auto_push"
git push


cd "D:\05 GIT\05_Python_projects"
git pull
git add .
git commit -m "05_Python_projects_auto_push"
git push


cd "D:\05 GIT\06_Statistical_Analysis_projects"
git pull
git add .
git commit -m "06_Statistical_Analysis_projects_auto_push"
git push


cd "D:\05 GIT\07_EDA_projects"
git pull
git add .
git commit -m "07_EDA_projects_auto_push"
git push


cd "D:\05 GIT\08_WebScraping_projects"
git pull
git add .
git commit -m "08_WebScraping_projects_auto_push"
git push


cd "D:\05 GIT\09_ML_projects"
git pull
git add .
git commit -m "09_ML_projects_auto_push"
git push


cd "D:\05 GIT\10_DL_projects"
git pull
git add .
git commit -m "10_DL_projects_auto_push"
git push


cd "D:\05 GIT\11_NLP_projects"
git pull
git add .
git commit -m "11_NLP_projects_auto_push"
git push


cd "D:\05 GIT\12_GenAI_projects"
git pull
git add .
git commit -m "12_GenAI_projects_auto_push"
git push


cd "D:\05 GIT\13_Misc_projects"
git pull
git add .
git commit -m "13_Misc_projects_auto_push"
git push


cd "D:\05 GIT\14_Proj_ML_99Acres"
git pull
git add .
git commit -m "14_Proj_ML_99Acres_auto_push"
git push

pause

exit