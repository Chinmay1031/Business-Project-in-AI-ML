import math
import numpy as np
import pickle
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime
import os
from huggingface_hub import hf_hub_download
import joblib

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load ML model from Hugging Face Hub
try:
    repo_id = "chinmay1031/ipl-model"  # ⬅️ Replace with your actual Hugging Face username/repo
    filename = "ml_model.pkl"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = joblib.load(model_path)
    print(f"✓ Model loaded successfully from Hugging Face: {repo_id}/{filename}")
except Exception as e:
    print("✗ Failed to load model from Hugging Face.")
    print("Error:", e)
    exit("Aborting app launch due to missing model.")


# Load the ML model
# Choose ONE of these options based on where your ml_model.pkl file is located:

# Option 1: If ml_model.pkl is in Documents folder
#filename = '//Users/chinmaybandekar/Documents/ml_model.pkl'

# Option 2: If ml_model.pkl is in the same folder as this app.py
# filename = 'ml_model.pkl'

# Option 3: If ml_model.pkl is in your project folder
# filename = '/Users/chinmaybandekar/Documents/Projects/Business-Project-in-AI-ML-1/ml_model.pkl'

# Option 4: Using expanduser (works for any user)
# filename = os.path.expanduser('~/Documents/ml_model.pkl')

#try:
    #model = pickle.load(open(filename, 'rb'))
   # print(f"✓ Model loaded successfully from: {filename}")
#except FileNotFoundError:
    #print(f"✗ Error: Could not find model file at: {filename}")
    #print("Please update the 'filename' variable with the correct path to your ml_model.pkl file")
    #exit()

# Define team list with updated names
teams = [
    'Chennai Super Kings',
    'Delhi Capitals',  # Updated from Delhi Daredevils
    'Punjab Kings',  # Updated from Kings XI Punjab
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad',
    'Gujarat Titans',
    'Lucknow Super Giants'
]

# Team colors for better visualizationnss
team_colors = {
    'Chennai Super Kings': '#FDB913',
    'Delhi Capitals': '#282968',
    'Punjab Kings': '#ED1B24',
    'Kolkata Knight Riders': '#3A225D',
    'Mumbai Indians': '#004BA0',
    'Rajasthan Royals': '#EA1A85',
    'Royal Challengers Bangalore': '#EC1C24',
    'Sunrisers Hyderabad': '#FF822A',
    'Gujarat Titans': '#1C2C3C',
    'Lucknow Super Giants': '#4E91CC'
}

# App layout with improved UI
app.layout = html.Div([
    # Background style
    html.Div([
        # Header with icon
        html.Div([
            html.H1([
                html.I(className="fas fa-cricket-ball", style={'margin-right': '15px'}),
                'ML POWERED LIVE SCORE PREDICTOR'
            ], style={
                'text-align': 'center',
                'color': 'white',
                'padding': '30px 20px',
                'margin': '0',
                'text-shadow': '2px 2px 4px rgba(0,0,0,0.5)',
                'font-weight': '700',
                'letter-spacing': '1px'
            })
        ], style={
            'background': 'linear-gradient(135deg, rgba(0,0,0,0.7), rgba(0,0,0,0.5))',
            'border-bottom': '3px solid #28a745',
            'margin-bottom': '20px'
        }),
        
        # Main content container with card design
        dbc.Container([
            dbc.Card([
                dbc.CardBody([
                    # Info Alert
                    dbc.Alert([
                        html.I(className="fas fa-info-circle", style={'margin-right': '10px'}),
                        "Minimun 5 overs data required for accurate prediction !"
                    ], color="info", dismissable=True, style={'margin-bottom': '25px'}),
                    
                    # Team Selection Section
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-users", style={'margin-right': '10px'}),
                            'Team Selection'
                        ], style={'color': '#333', 'margin-bottom': '20px'}),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label([
                                    html.I(className="fas fa-baseball-bat-ball", style={'margin-right': '8px'}),
                                    'Select the Batting Team'
                                ], style={'font-weight': 'bold', 'color': '#555', 'margin-bottom': '8px'}),
                                dcc.Dropdown(
                                    id='batting-team',
                                    options=[{'label': team, 'value': team} for team in teams],
                                    value='Chennai Super Kings',
                                    clearable=False,
                                    style={'margin-bottom': '20px'}
                                )
                            ], md=6),
                            
                            dbc.Col([
                                html.Label([
                                    html.I(className="fas fa-bowling-ball", style={'margin-right': '8px'}),
                                    'Select the Bowling Team'
                                ], style={'font-weight': 'bold', 'color': '#555', 'margin-bottom': '8px'}),
                                dcc.Dropdown(
                                    id='bowling-team',
                                    options=[{'label': team, 'value': team} for team in teams],
                                    value='Mumbai Indians',
                                    clearable=False,
                                    style={'margin-bottom': '20px'}
                                )
                            ], md=6)
                        ]),
                        
                        # Error message for same team selection
                        html.Div(id='team-error', style={'margin-bottom': '20px'})
                    ], style={
                        'background': '#f8f9fa',
                        'padding': '25px',
                        'border-radius': '10px',
                        'margin-bottom': '25px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
                    }),
                    
                    # Match Statistics Section
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-chart-line", style={'margin-right': '10px'}),
                            'Live Match Statistics'
                        ], style={'color': '#333', 'margin-bottom': '20px'}),
                        
                        # Overs and Runs
                        dbc.Row([
                            dbc.Col([
                                html.Label([
                                    html.I(className="fas fa-clock", style={'margin-right': '8px'}),
                                    'Current Over'
                                ], style={'font-weight': 'bold', 'color': '#555', 'margin-bottom': '8px'}),
                                dcc.Input(
                                    id='overs',
                                    type='number',
                                    min=5.1,
                                    max=19.5,
                                    step=0.1,
                                    value=5.1,
                                    style={
                                        'width': '100%',
                                        'padding': '10px',
                                        'border': '2px solid #ddd',
                                        'border-radius': '5px',
                                        'font-size': '16px'
                                    }
                                ),
                                html.Div(id='overs-error', style={'margin-top': '8px'})
                            ], md=6),
                            
                            dbc.Col([
                                html.Label([
                                    html.I(className="fas fa-calculator", style={'margin-right': '8px'}),
                                    'Current Runs'
                                ], style={'font-weight': 'bold', 'color': '#555', 'margin-bottom': '8px'}),
                                dcc.Input(
                                    id='runs',
                                    type='number',
                                    min=0,
                                    max=354,
                                    step=1,
                                    value=0,
                                    style={
                                        'width': '100%',
                                        'padding': '10px',
                                        'border': '2px solid #ddd',
                                        'border-radius': '5px',
                                        'font-size': '16px'
                                    }
                                )
                            ], md=6)
                        ], style={'margin-bottom': '25px'}),
                        
                        # Wickets Fallen
                        html.Div([
                            html.Label([
                                html.I(className="fas fa-exclamation-triangle", style={'margin-right': '8px'}),
                                'Wickets Fallen'
                            ], style={'font-weight': 'bold', 'color': '#555', 'margin-bottom': '15px'}),
                            dcc.Slider(
                                id='wickets',
                                min=0,
                                max=9,
                                step=1,
                                value=0,
                                marks={i: {'label': str(i), 'style': {'color': '#666', 'font-weight': 'bold'}} for i in range(10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'margin-bottom': '30px'}),
                        
                        # Last 5 Overs Stats
                        html.H5([
                            html.I(className="fas fa-history", style={'margin-right': '10px'}),
                            'Last 5 Overs Performance'
                        ], style={'color': '#555', 'margin-bottom': '15px', 'margin-top': '10px'}),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label([
                                    html.I(className="fas fa-running", style={'margin-right': '8px'}),
                                    'Runs Scored'
                                ], style={'font-weight': 'bold', 'color': '#555', 'margin-bottom': '8px'}),
                                dcc.Input(
                                    id='runs-prev-5',
                                    type='number',
                                    min=0,
                                    step=1,
                                    value=0,
                                    style={
                                        'width': '100%',
                                        'padding': '10px',
                                        'border': '2px solid #ddd',
                                        'border-radius': '5px',
                                        'font-size': '16px'
                                    }
                                )
                            ], md=6),
                            
                            dbc.Col([
                                html.Label([
                                    html.I(className="fas fa-times-circle", style={'margin-right': '8px'}),
                                    'Wickets Taken'
                                ], style={'font-weight': 'bold', 'color': '#555', 'margin-bottom': '8px'}),
                                dcc.Input(
                                    id='wickets-prev-5',
                                    type='number',
                                    min=0,
                                    step=1,
                                    value=0,
                                    style={
                                        'width': '100%',
                                        'padding': '10px',
                                        'border': '2px solid #ddd',
                                        'border-radius': '5px',
                                        'font-size': '16px'
                                    }
                                )
                            ], md=6)
                        ])
                    ], style={
                        'background': '#f8f9fa',
                        'padding': '25px',
                        'border-radius': '10px',
                        'margin-bottom': '25px',
                        'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
                    }),
                    
                    # Predict Button with improved styling
                    dbc.Button([
                        html.I(className="fas fa-magic", style={'margin-right': '10px'}),
                        'Predict Score'
                    ],
                        id='predict-button',
                        n_clicks=0,
                        color="success",
                        size="lg",
                        style={
                            'width': '100%',
                            'padding': '15px',
                            'font-size': '20px',
                            'font-weight': 'bold',
                            'border-radius': '8px',
                            'margin-bottom': '25px',
                            'box-shadow': '0 4px 6px rgba(0,0,0,0.1)',
                            'transition': 'all 0.3s ease'
                        }
                    ),
                    
                    # Prediction Result
                    html.Div(id='prediction-output'),
                    
                    # Additional Stats Display
                    html.Div(id='match-stats', style={'margin-top': '20px'})
                ])
            ], style={
                'background': 'rgba(255, 255, 255, 0.95)',
                'border-radius': '15px',
                'box-shadow': '0 10px 30px rgba(0,0,0,0.3)',
                'border': 'none'
            })
        ], fluid=True, style={'max-width': '900px', 'padding': '20px'}),
        
        # Footer
        html.Div([
            html.P([
                html.I(className="fas fa-code", style={'margin-right': '8px'}),
                f'Powered by Machine Learning | © {datetime.now().year}'
            ], style={
                'text-align': 'center',
                'color': 'white',
                'margin': '0',
                'padding': '15px',
                'font-size': '14px'
            })
        ], style={
            'background': 'rgba(0,0,0,0.6)',
            'margin-top': '30px'
        })
        
    ], style={
        'background-color': '#f0f2f5',
        'background-image': 'url("/assets/background.jpg")',
        'background-attachment': 'fixed',
        'background-size': 'cover',
        'min-height': '100vh',
        'padding': '0'
    }),
    
    # Add Font Awesome for icons
    html.Link(
        rel='stylesheet',
        href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
    )
])


def get_team_encoding(team_name):
    """Convert team name to one-hot encoding"""
    encoding = [0, 0, 0, 0, 0, 0, 0, 0]
    # Map new team names to old encoding
    team_map = {
        'Chennai Super Kings': 'Chennai Super Kings',
        'Delhi Capitals': 'Delhi Daredevils',
        'Punjab Kings': 'Kings XI Punjab',
        'Kolkata Knight Riders': 'Kolkata Knight Riders',
        'Mumbai Indians': 'Mumbai Indians',
        'Rajasthan Royals': 'Rajasthan Royals',
        'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
        'Sunrisers Hyderabad': 'Sunrisers Hyderabad',
        'Gujarat Titans': 'Mumbai Indians',  # Map to similar team
        'Lucknow Super Giants': 'Kings XI Punjab'  # Map to similar team
    }
    
    team_index = {
        'Chennai Super Kings': 0,
        'Delhi Daredevils': 1,
        'Kings XI Punjab': 2,
        'Kolkata Knight Riders': 3,
        'Mumbai Indians': 4,
        'Rajasthan Royals': 5,
        'Royal Challengers Bangalore': 6,
        'Sunrisers Hyderabad': 7
    }
    
    mapped_team = team_map.get(team_name, team_name)
    if mapped_team in team_index:
        encoding[team_index[mapped_team]] = 1
    return encoding


# Callback for team validation
@app.callback(
    Output('team-error', 'children'),
    Input('batting-team', 'value'),
    Input('bowling-team', 'value')
)
def validate_teams(batting_team, bowling_team):
    if batting_team == bowling_team:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-circle", style={'margin-right': '10px'}),
            'Error: Batting and Bowling teams must be different!'
        ],
            color='danger',
            style={'margin-top': '10px'}
        )
    return None


# Callback for overs validation
@app.callback(
    Output('overs-error', 'children'),
    Input('overs', 'value')
)
def validate_overs(overs):
    if overs is not None:
        decimal_part = overs - math.floor(overs)
        if decimal_part > 0.5:
            return html.Small([
                html.I(className="fas fa-exclamation-triangle", style={'margin-right': '5px'}),
                'Invalid over! Each over has only 6 balls (0.0 to 0.5)'
            ], style={'color': '#dc3545', 'font-weight': 'bold'})
    return None


# Callback for prediction and stats
@app.callback(
    [Output('prediction-output', 'children'),
     Output('match-stats', 'children')],
    Input('predict-button', 'n_clicks'),
    State('batting-team', 'value'),
    State('bowling-team', 'value'),
    State('overs', 'value'),
    State('runs', 'value'),
    State('wickets', 'value'),
    State('runs-prev-5', 'value'),
    State('wickets-prev-5', 'value')
)
def predict_score(n_clicks, batting_team, bowling_team, overs, runs, wickets, runs_prev_5, wickets_prev_5):
    if n_clicks == 0:
        return None, None
    
    # Validate inputs
    if batting_team == bowling_team:
        return dbc.Alert([
            html.I(className="fas fa-times-circle", style={'margin-right': '10px'}),
            'Please select different teams for batting and bowling'
        ], color='warning'), None
    
    if overs is not None:
        decimal_part = overs - math.floor(overs)
        if decimal_part > 0.5:
            return dbc.Alert([
                html.I(className="fas fa-times-circle", style={'margin-right': '10px'}),
                'Please enter a valid over value'
            ], color='warning'), None
    
    # Build prediction array
    prediction_array = []
    prediction_array.extend(get_team_encoding(batting_team))
    prediction_array.extend(get_team_encoding(bowling_team))
    prediction_array.extend([runs, wickets, overs, runs_prev_5, wickets_prev_5])
    
    # Convert to numpy array and predict
    prediction_array = np.array([prediction_array])
    predict = model.predict(prediction_array)
    my_prediction = int(round(predict[0]))
    
    # Calculate additional stats
    balls_remaining = (20 - overs) * 6
    current_run_rate = runs / overs if overs > 0 else 0
    required_run_rate = (my_prediction - runs) / ((20 - overs) if overs < 20 else 1)
    
    # Prediction result with enhanced design
    prediction_result = dbc.Card([
        dbc.CardBody([
            html.Div([
                html.I(className="fas fa-trophy", style={
                    'font-size': '40px',
                    'color': '#ffd700',
                    'margin-bottom': '15px'
                }),
                html.H2('PREDICTED FINAL SCORE', style={
                    'color': '#28a745',
                    'font-weight': 'bold',
                    'margin-bottom': '10px'
                }),
                html.H1(f'{my_prediction - 5} - {my_prediction + 5}', style={
                    'color': '#333',
                    'font-weight': 'bold',
                    'font-size': '48px',
                    'margin': '20px 0'
                }),
                html.P(f'Most Likely Score: {my_prediction}', style={
                    'color': '#666',
                    'font-size': '18px',
                    'font-style': 'italic'
                })
            ], style={'text-align': 'center'})
        ])
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'background': 'white',
        'border': '3px solid #28a745',
        'border-radius': '15px',
        'box-shadow': '0 8px 16px rgba(0,0,0,0.2)',
        'animation': 'fadeIn 0.5s'
    })
    
    # Additional match statistics
    stats_cards = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.I(className="fas fa-tachometer-alt", style={
                        'font-size': '30px',
                        'color': '#17a2b8',
                        'margin-bottom': '10px'
                    }),
                    html.H5('Current Run Rate', style={'color': '#666', 'margin-bottom': '5px'}),
                    html.H3(f'{current_run_rate:.2f}', style={'color': '#333', 'font-weight': 'bold'})
                ], style={'text-align': 'center', 'padding': '20px'})
            ], style={'border-radius': '10px', 'box-shadow': '0 2px 8px rgba(0,0,0,0.1)'})
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.I(className="fas fa-bullseye", style={
                        'font-size': '30px',
                        'color': '#ffc107',
                        'margin-bottom': '10px'
                    }),
                    html.H5('Required Run Rate', style={'color': '#666', 'margin-bottom': '5px'}),
                    html.H3(f'{required_run_rate:.2f}', style={'color': '#333', 'font-weight': 'bold'})
                ], style={'text-align': 'center', 'padding': '20px'})
            ], style={'border-radius': '10px', 'box-shadow': '0 2px 8px rgba(0,0,0,0.1)'})
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.I(className="fas fa-baseball-ball", style={
                        'font-size': '30px',
                        'color': '#dc3545',
                        'margin-bottom': '10px'
                    }),
                    html.H5('Balls Remaining', style={'color': '#666', 'margin-bottom': '5px'}),
                    html.H3(f'{int(balls_remaining)}', style={'color': '#333', 'font-weight': 'bold'})
                ], style={'text-align': 'center', 'padding': '20px'})
            ], style={'border-radius': '10px', 'box-shadow': '0 2px 8px rgba(0,0,0,0.1)'})
        ], md=4)
    ])
    
    return prediction_result, stats_cards


if __name__ == '__main__':
    app.run(debug=True)