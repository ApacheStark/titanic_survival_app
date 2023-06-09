import dash
from dash import dcc, html
import base64

import pickle

from sklearn.ensemble import RandomForestClassifier

import pandas as pd 

# python app3.py 
# server running on 127.0.0.1:8050

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Would You Survive the Titanic Sinking?"

button_style = {
    "width": "240px",
    "height": "40px",
    # "cursor": "pointer",
    "border": "10px",
    "border-radius": "5px",
    # "background-color": "white",
    # "color": "white",
    # "text-transform": "uppercase",
    "font-size": "12px",
    "margin": "5px"
  }


app.layout = html.Div([
    html.H1("Would You Survive the Titanic Sinking?"),
    html.Hr(),
    html.P("Test your chances of survival given your name, age, gender, how many you invite, and where you stay!"),
    html.Hr(),
    dcc.Input(id="Name", type="text", placeholder="Name?", style = button_style),
    html.Hr(),
    dcc.Input(id="Age", type="number", placeholder="Age?", style = button_style),
    html.Hr(),
    dcc.Input(id="Gender", type="text", placeholder="Gender?", style = button_style),
    html.Hr(),
    dcc.Input(id="Fam_Size", type="number", placeholder="How many do you invite?", style = button_style),
    html.Hr(),
    dcc.Dropdown(['S', 'A', 'B', 'C', 'D','E','F','G','Working'], '', id='Ticket', placeholder='Ticket? (refer to diagram below)', 
    style = button_style),
    html.Hr(),
    html.Button("What are my chances?", id="predict"),
    html.Div(id="output"),
    html.Hr(),
    html.Img(src=r'assets/cabin_ref.png', alt='image')    
])

@app.callback(
    
    dash.dependencies.Output("output", "children"),
    
    [dash.dependencies.Input("predict", "n_clicks")],
    
    [dash.dependencies.State("Name", "value"),
     dash.dependencies.State("Age", "value"),
     dash.dependencies.State("Gender", "value"),
     dash.dependencies.State("Fam_Size", "value"),
     dash.dependencies.State("Ticket", "value")]
    
    )


def do_anything(n_clicks, Name, Age, Gender, Fam_Size, Ticket):
    
    if Name is not None and Age is not None and Gender is not None and Fam_Size is not None and Ticket is not None:

        def if_fem(x):
            x = str(x).lower()
            if x == 'f' or x == 'woman' or 'fem' in x:
                return 1
            else:
                return 0

        def if_mal(x):
            x = str(x).lower()
            if x == 'male' or x == 'm' or x == 'man' or 'masc' in x:
                return 1
            else:
                return 0 

        def ticket_pclass1(x):
            if x in ['S', 'A','B']:
                return 1
            else:
                return 0

        def ticket_pclass2(x):
            if x in ['B', 'C','D','E']:
                return 1
            else:
                return 0

        def ticket_pclass3(x):
            if x in ['E', 'F','G']:
                return 1
            else:
                return 0

        input_df = pd.DataFrame.from_dict(
                    data = {
                        'Age': [int(Age)],
                        'Fam_Size': [int(Fam_Size)],
                        'gender': [Gender],
                        'ticket': [Ticket]
                    }
                )

        
        Name_len = len(Name)
        input_df['Name_len'] = Name_len

        input_df['female'] = input_df.gender.apply(if_fem)
        input_df['male'] = input_df.gender.apply(if_mal)

        input_df['1_class'] = input_df.ticket.apply(ticket_pclass1)
        input_df['2_class'] = input_df.ticket.apply(ticket_pclass2)
        input_df['3_class'] = input_df.ticket.apply(ticket_pclass3)

        alt_df = input_df.copy()

        if alt_df['Fam_Size'][0] > 0:
            alt_df['Fam_Size'] = 0
        else:
            alt_df['Fam_Size'] = 1

        # keep_cols = pickle.load(open('model_columns.pkl', 'rb'))
        keep_cols = ['Age', 'Fam_Size', 'Name_len', 'female', 'male', '1_class', '2_class', '3_class']

        clf = pickle.load(open('model.pkl', 'rb'))

        chance = clf.predict_proba(input_df[keep_cols])[0][1]
        p_chance = round(chance * 100, 1)

        alt_chance = clf.predict_proba(alt_df[keep_cols])[0][1]
        alt_p_chance = round(alt_chance * 100, 1)

        if alt_p_chance > p_chance and alt_df['Fam_Size'][0] == 0: 
            
            alt_string = f"If you bought no one with you, it was {alt_p_chance}%, they're holding you back!"
        
        elif alt_p_chance > p_chance and alt_df['Fam_Size'][0] == 1:
            
            alt_string = f"If you bought at least 1 person with you, it was {alt_p_chance}%, you needed help!"
        
        elif alt_p_chance < p_chance and alt_df['Fam_Size'][0] == 0:
            
            alt_string = f"If you bought no one with you, it was {alt_p_chance}%, lucky you had someone!"
        
        elif alt_p_chance < p_chance and alt_df['Fam_Size'][0] == 1:
            
            alt_string = f"If you bought at least 1 person with you, it was {alt_p_chance}%, lucky you went alone!"
        
        else:
            alt_string = ''

        return f"You are {Name} who is {Age} years old, staying on level {Ticket}.\nYou bought along {Fam_Size} friends and family. You have a {p_chance}% chance of survival. " + alt_string 
    else:
        return ""

if __name__ == '__main__':
    app.run_server(debug=True)
