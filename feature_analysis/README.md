# Feature Analysis

This program is meant to be used to analyze the feature importance of any dataset.

## How to Run This Program

To run this program, I have created a pipeline that simply takes in the dataset and will return a dictionary of importance scores. Run `python3 main.py your_dataset.csv` to obtain results. There are other commandline options available for greater control. For more informatin, run `python3 main.py -h`. 

IMPORTANT: The target label MUST be in the *first* column of the dataset. The program may still run, but it will not run with the right columns as features. Also, remember that sklearn models make up the basis of this program. Most of them cannot take NaN or object (aka string) values as input. If you see these types of errors arise, adjust the dataset accordingly. 

NOTE: You probably want to pay special attention to the `-f` argument. This program can either run decently fast or very, very slow depending on how much you care about importance scores. Without `-f`, the whole ensemble runs, including some very slow models.  However, if you don't need the most precise scoring possible, then running with `-f` removes those slower models and significantly speeds up run time.  

## Output

The output from this program is a dictionary of importance scores either printed to the terminal or to a json file, which can be specified via the commandline arguments. The bigger the importance score, the more important the feature is expected to be. Output is ordered from most important -> least important. 

Feel free to tweak your copy of the code, but please let me know if there is functionality you want! I can probably make it happen, and you won't have to do it manually.  The whole idea of this program is ease of use and a limited amount of steps to achieve the best results. More importantly, let me know if you run into any bugs. This program is still in its test phase, and the user is the best person to figure out where things don't work or how things could work better.

As always, feel free to reach out to me via Teams or nrubocki@naturalintelligence.ai. 
