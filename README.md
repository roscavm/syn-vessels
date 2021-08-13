Run get_voyages_main in get_voyages.py to get temporary processed files. Use the temporary processed files as input for get_prediction_main in get_prediction.py to get voyages and vessels csv outputs

get_prediction_main accepts prediction='rf' to train a random forest model for the 2nd and 3rd predictions; or prediction ='standard' for a simple predictor based on past voyages