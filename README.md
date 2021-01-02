# MillionSongRecommendations_Collaborativefiltering_PySpark
This project is using collaborative filtering to do the product recommendations.

## Purpose and process
1. Data processing: to know the data structure and data history in these million songs datasets and generate the dataframes (song features, song play records, user play songs records) will be used to train the model.
2. Audio similarity: Clean the data, song features correlation analysis, solve the unbalanced song genre classification problem
3. Classification model selection: evaluate different models perfromance by using confusion matrix, precision, recall, accuracy and AUROC.
4. Song recommendations: Combine the song play records and user play records to do collaborative filtering, using Alternating Least Squares(ALS) to train the implicit matric factorization model, which will generate songs recommendations for specific users based on the combined user-song play information. 
5. Model evaluation: The evaluation matrix are Precision (@ 5), NDCG (@ 10), Mean Average Precision (MAP)(@ 5 or 10).

## Data
Million Song Dataset (MSD) is an integrated database of Million Songs (size 13.9G). This project initiated by The Echo Nest and LabROSA. There are other 7 datasets are involved in the this study. However, in this report, it will focus on the MSD, Taste Profile and Top MAGD datasets. It mainly discuss the audio similarity by using machine learning model to classify the genre of songs and try to do the song recommendations.

The data and more information, please refer to http://millionsongdataset.com

## Output
The report is based on the assignment requirement. In the report, it does not mention the question, which may make you confused.
The pipeline may not the best try. It is welcome to add more advanced solutions.


In this repo, I did not add the original datasets (too large). The Spark config set up is also not included. Please help yourself!

Author: Shi Chen
E-mail:yuejianyingmm@icloud.com
