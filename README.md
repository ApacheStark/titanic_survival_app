# Would You Survive The Titanic?

Utilises:
- Python
- Dash
- Pandas
- Scikit-Learn
- PythonAnywhere

Test your chances of surviving the titanic given who you are, who you brought along, and where you are bunking.

The app uses the dash framework, predicting with a random forest classifier to produce a probability of survival, this model was pickled as model.pkl and recalled in the app, no update learning occurs.  

The %wsgi.py file contains the python code to run the actual server, in this instance it grabs the app object in the "flask_app.py" on pythonanywhere and creates a server object as called 'application' (it must recognise that key word is the server when deploying).

The assets folder is again a default for dash where local images and css files are assumed to live.

The pythonanywhere environment has many python libraries preinstalled (dash, pandas, scikit, xgboost, etc), therefore little to no need to create a virtual environment.
