import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import pandas as pd
import numpy as np
import pycountry
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv #To load the .env file


ct = pd.read_csv(r'csv')
ct.head()
API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
 
)




# colours
backgroundcolor = "#0c0414"
supplierc = "#8E24AA"
recipientc = "#00bbbe"
fontcolor = "#ffffff"
red = '#ff0003'
color_discrete_map = {'Actual': '#ff0003', 'Forecasted': 'blue'}


# Function to get ISO Alpha-3 and ISO Numeric codes
def get_iso_codes(country_name):
    try:
        country = pycountry.countries.lookup(country_name)
        return pd.Series([country.alpha_3, country.numeric])
    except LookupError:
        return pd.Series([None, None])  # Handles missing countries


ct['Delivery year'] = pd.to_numeric(ct['Delivery year'], errors='coerce').fillna(0).astype(int)

# Apply the function to Supplier and Recipient columns
ct[['Supplier_ISO_Alpha_3', 'Supplier_ISO_Numeric']] = ct['Supplier'].apply(get_iso_codes)
ct[['Recipient_ISO_Alpha_3', 'Recipient_ISO_Numeric']] = ct['Recipient'].apply(get_iso_codes)

# Manually setting ISO codes for specific countries
ct.loc[ct['Supplier'].str.strip() == 'Russia', 'Supplier_ISO_Alpha_3'] = 'RUS'
ct.loc[ct['Supplier'].str.strip() == 'Russia', 'Supplier_ISO_Numeric'] = '643'

ct.loc[ct['Supplier'].str.strip() == 'Turkiye', 'Supplier_ISO_Alpha_3'] = 'TUR'
ct.loc[ct['Supplier'].str.strip() == 'Turkiye', 'Supplier_ISO_Numeric'] = '792'

ct.loc[ct['Supplier'].str.strip() == 'Bosnia-Herzegovina', 'Supplier_ISO_Alpha_3'] = 'BIH'
ct.loc[ct['Supplier'].str.strip() == 'Bosnia-Herzegovina', 'Supplier_ISO_Numeric'] = '070'

ct.loc[ct['Recipient'].str.strip() == 'Bosnia-Herzegovina', 'Recipient_ISO_Alpha_3'] = 'BIH'
ct.loc[ct['Recipient'].str.strip() == 'Bosnia-Herzegovina', 'Recipient_ISO_Numeric'] = '070'

ct.loc[ct['Supplier'].str.strip() == 'Brunei',
       ['Supplier_ISO_Alpha_3', 'Supplier_ISO_Numeric']] = ['BRN', '096']

ct.loc[ct['Recipient'].str.strip() == 'Brunei',
       ['Recipient_ISO_Alpha_3', 'Recipient_ISO_Numeric']] = ['BRN', '096']

ct.loc[ct['Supplier'].str.strip() == "Cote d'Ivoire",
       ['Supplier_ISO_Alpha_3', 'Supplier_ISO_Numeric']] = ['CIV', '384']

ct.loc[ct['Recipient'].str.strip() == "Cote d'Ivoire",
       ['Recipient_ISO_Alpha_3', 'Recipient_ISO_Numeric']] = ['CIV', '384']

ct.loc[ct['Supplier'].str.strip() == 'Libya',
       ['Supplier_ISO_Alpha_3', 'Supplier_ISO_Numeric']] = ['LBY', '434']
ct.loc[ct['Recipient'].str.strip() == 'Libya',
       ['Recipient_ISO_Alpha_3', 'Recipient_ISO_Numeric']] = ['LBY', '434']

# Assign ISO codes for Sudan
ct.loc[ct['Supplier'].str.strip() == 'Sudan',
       ['Supplier_ISO_Alpha_3', 'Supplier_ISO_Numeric']] = ['SDN', '729']
ct.loc[ct['Recipient'].str.strip() == 'Sudan',
       ['Recipient_ISO_Alpha_3', 'Recipient_ISO_Numeric']] = ['SDN', '729']

# Assign ISO codes for DR Congo
ct.loc[ct['Supplier'].str.strip() == 'DR Congo',
       ['Supplier_ISO_Alpha_3', 'Supplier_ISO_Numeric']] = ['COD', '180']
ct.loc[ct['Recipient'].str.strip() == 'DR Congo',
       ['Recipient_ISO_Alpha_3', 'Recipient_ISO_Numeric']] = ['COD', '180']

# Assign ISO codes for Palestine
ct.loc[ct['Recipient'].str.strip() == 'Palestine',
       ['Recipient_ISO_Alpha_3', 'Recipient_ISO_Numeric']] = ['PSE', '275']

# Assign ISO codes for Kosovo
ct.loc[ct['Recipient'].str.strip() == 'Kosovo',
       ['Recipient_ISO_Alpha_3', 'Recipient_ISO_Numeric']] = ['XKX', '999']

# Assign ISO codes for Libya
ct.loc[ct['Recipient'].str.strip() == 'Libya',
       ['Recipient_ISO_Alpha_3', 'Recipient_ISO_Numeric']] = ['LBY', '434']

# renaming UAE in supplers column to Uited Arab Emirates
ct['Supplier'] = ct['Supplier'].replace('UAE', 'United Arab Emirates')




# correct Lybia Spelling
ct['Recipient'] = ct['Recipient'].replace('Lybia', 'Libya')

ct['Recipient'] = ct['Recipient'].replace('NATO**','NATO')


# Ensuring that time_series_data uses .loc
time_series_data = ct[['Delivery year', 'Numbers delivered']]
time_series_data.loc[:, 'Delivery year'] = pd.to_datetime(time_series_data['Delivery year'], format='%Y')
time_series_data = time_series_data.groupby('Delivery year')['Numbers delivered'].sum().reset_index()
time_series_data.columns = ['ds', 'y']

# Prophet Model
from prophet import Prophet
model = Prophet()
model.fit(time_series_data)

# Predict future values
future = model.make_future_dataframe(periods=3, freq='A')
forecast = model.predict(future)

# Extract forecast and process columns
forecast_trimmed = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast_trimmed.rename(columns={'yhat': 'Predicted Numbers', 'ds': 'Delivery year'}, inplace=True)
forecast_trimmed['Delivery year'] = pd.to_datetime(forecast_trimmed['Delivery year'], errors='coerce').dt.year





# Creating the geo dataframe
geo = pd.concat([
    ct[['Supplier', 'Supplier_ISO_Alpha_3', 'Order date', 'Delivery year', 'Numbers delivered', 'Status', 
        'Recipient Affiliation', 'Armament category']].rename(columns={
        'Supplier': 'Country', 
        'Supplier_ISO_Alpha_3': 'ISO_Code'
    }).assign(Role='Supplier'),
    
    ct[['Recipient', 'Recipient_ISO_Alpha_3', 'Order date', 'Delivery year', 'Numbers delivered', 'Status', 
        'Recipient Affiliation', 'Armament category']].rename(columns={
        'Recipient': 'Country', 
        'Recipient_ISO_Alpha_3': 'ISO_Code'
    }).assign(Role='Recipient', 
              Delivery_year='',  # Set Delivery_year to blank for recipients
              Numbers_delivered='')  # Set Numbers_delivered to blank for recipients
], ignore_index=True).dropna(subset=['ISO_Code'])



forecast_trimmed['Delivery year'] = forecast_trimmed['Delivery year'].astype(int)

geo = pd.merge(geo, forecast_trimmed[['Delivery year', 'Predicted Numbers']], 
               on='Delivery year', how='left')


# map
map_figure = px.scatter_geo(
    geo,
    locations='ISO_Code',
    projection='orthographic',
    color='Role',
    opacity=0.5,
    hover_name='Country',
    hover_data=['Order date', 'Numbers delivered', 'Predicted Numbers', 'Status', 
                'Recipient Affiliation', 'Armament category'],
    color_discrete_sequence=['#FF5733', '#33FF57']
)
map_figure.update_geos(
    landcolor='lightgray',
    oceancolor='lightblue',
    coastlinecolor='darkblue',
    showland=True,
    showocean=True,
    showcoastlines=True
)
map_figure.update_layout(
    title={'text': 'Global Arms Transfers: Supplier and Recipient Map', 'x': 0.5},
    plot_bgcolor=backgroundcolor,
    paper_bgcolor=backgroundcolor,
    font_color=fontcolor,
    geo=dict(bgcolor=backgroundcolor, showframe=False, coastlinecolor=fontcolor)
)


# Supplied and received over the years
sup_rec = ct[['Supplier', 'Recipient', 'Numbers delivered', 'Delivery year']]

# Melt sup_rec
sup_rec_melt = sup_rec.melt(id_vars=['Delivery year', 'Numbers delivered'],
                            value_vars=['Supplier', 'Recipient'],
                            var_name='Role')

# Aggregate the sup_rec_melt
sup_rec_melt = sup_rec_melt.groupby(['Delivery year', 'value', 'Role'])['Numbers delivered'].sum().reset_index()

# Incorporating forecasted data into the line chart
# Combine forecast data with the existing dataset
forecast_trimmed.loc[:, 'Delivery year'] = forecast_trimmed['Delivery year'].astype(int)  # Ensure 'Delivery year' is integer
sup_rec_melt = pd.merge(
    sup_rec_melt, 
    forecast_trimmed[['Delivery year', 'Predicted Numbers']], 
    on='Delivery year', 
    how='outer'
)

# Fill missing values in 'Numbers delivered' for future years
sup_rec_melt['Numbers delivered'] = sup_rec_melt['Numbers delivered'].fillna(0)  # Replace NaN with 0

print(sup_rec_melt)

# Line chart creation
line_fig = px.line(
    sup_rec_melt, 
    x='Delivery year', 
    y='Numbers delivered', 
    color='Role',  # Differentiating Supplier and Recipient roles
    color_discrete_sequence=['#ff0003']
)

# Adding predicted data to the hover information
line_fig.add_scatter(
    x=sup_rec_melt['Delivery year'], 
    y=sup_rec_melt['Predicted Numbers'], 
    mode='lines', 
    name='Forecast', 
    line=dict(dash='dash', color='blue')  # Style forecast line as dashed blue
)

# Chart layout customization
line_fig.update_layout(
    title={
        'text': 'Supply and Receipt of Armaments Over Time',
        'xanchor': 'center',
        'x': 0.5
    },
    plot_bgcolor=backgroundcolor,
    paper_bgcolor=backgroundcolor,
    font_color=fontcolor,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

# bar chart

ct_sort = ct[['Supplier','Recipient','Armament category', 'Numbers delivered','Delivery year']]

ct_sort = ct.sort_values(by=['Armament category', 'Numbers delivered'], ascending=[True, False])

ct_sort_melt = ct_sort.melt(id_vars = ['Armament category','Numbers delivered','Delivery year'],
                            value_vars = ['Supplier','Recipient'],
                            var_name = 'Role',
                            value_name ='Country',
                            )
ct_sort_melt = ct_sort_melt.sort_values(by= 'Numbers delivered', ascending= False)

ct_sort_melt = ct_sort_melt.groupby(['Armament category', 'Delivery year', 'Role', 'Country'])['Numbers delivered'].sum().reset_index()

# merging with forcasted model
ct_sort_melt = pd.merge(
    ct_sort_melt,
    forecast_trimmed[['Delivery year','Predicted Numbers']],
    on = 'Delivery year',
    how = 'outer'

)

# Reshape the DataFrame
ct_sort_melt = pd.melt(
    ct_sort_melt,
    id_vars=["Armament category", "Delivery year", "Role", "Country"],
    value_vars=["Numbers delivered", "Predicted Numbers"],
    var_name="Type",
    value_name="Value",
)





print(ct_sort_melt)




# Create the original bar chart for actual data
new_bar = px.bar(
    ct_sort_melt, 
    x='Armament category', 
    y='Value', 
    color = 'Type',
    barmode = 'group',
    title='Actual vs Forecasted Armament Deliveries',
    color_discrete_map= color_discrete_map,  # Set color for the actual data
)



# Customize the layout
new_bar.update_layout(
    title={
        'text': 'Total Armaments Supplied & Received (Including Forecast)',
        'xanchor': 'center',
        'yanchor': 'middle',
        'x': 0.5
    },
    plot_bgcolor=backgroundcolor,
    paper_bgcolor=backgroundcolor,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    font_color=fontcolor
)

print(ct_sort_melt['Type'].unique())

country_options = [{'label': country, 'value': iso_code} for country, iso_code in zip(geo['Country'], geo['ISO_Code'])]
# Remove duplicates
country_options = list({frozenset(option.items()): option for option in country_options}.values())








# App name
arms = dash.Dash(__name__)

# App layout
arms.layout = html.Div(
    style={
        'background-color': backgroundcolor,
        'height': '100vh',
        'width': '100vw',
        'display': 'flex',
        'flexDirection': 'column',              # Stack elements vertically
        'alignItems': 'center',                 # Center items horizontally
        'justifyContent': 'flex-start',         # Align items to the top
        'padding': '0'                          # No extra padding for the main container
    },
    children=[
        # Dropdown row
        html.Div(
            children=[
                # Dropdown 1
                dcc.Dropdown(
                    id='arms_dropdown',
                    options=[
                        {'label': 'Supplier', 'value': 'Supplier'},
                        {'label': 'Recipient', 'value': 'Recipient'}
                    ],
                    optionHeight=30,
                    value=None,
                    disabled=False,
                    multi=True,
                    searchable=True,
                    search_value='',
                    placeholder='Select supplier or recipient',
                    clearable=True,
                    style={
                        'width': '300px',               # Fixed width for dropdown
                        'background-color': backgroundcolor,
                        'color': '#000000',             # Text color
                        'padding': '10px',              # Padding inside dropdown
                        'font-size': '16px'             # Larger font
                    }
                ),
                # Space between dropdowns
                html.Div(style={'width': '50px'}),      # Spacer for gap
                # Dropdown 2
                dcc.Dropdown(
                    id='country_dropdown',
                    options=country_options,
                    optionHeight=30,
                    value=None,
                    disabled=False,
                    multi=True,
                    searchable=True,
                    search_value='',
                    placeholder='Select country',
                    clearable=True,
                    style={
                        'width': '300px',               # Fixed width for dropdown
                        'background-color': backgroundcolor,
                        'color': '#000000',             # Text color
                        'padding': '10px',              # Padding inside dropdown
                        'font-size': '16px'             # Larger font
                    }
                )
            ],
            style={
                'display': 'flex',                      # Arrange dropdowns in a row
                'flexDirection': 'row',                 # Horizontal alignment
                'alignItems': 'center',                 # Center-align items vertically
                'justifyContent': 'center',             # Center-align items horizontally
                'width': '100%',                        # Full width container
                'margin-top': '20px',                   # Space from the top
                'margin-bottom': '20px'                 # Space below dropdowns
            }
        ),
        # Charts row
        html.Div(
            children=[
                # Map
                html.Div(
                    children=dcc.Graph(
                        figure=map_figure,
                        id='map_1'
                    ),
                    style={
                        'flex': '1',                    # Equal space for all charts
                        'margin': '10px',               # Space around each chart
                        'background-color': backgroundcolor,
                        'padding': '10px'               # for Internal padding
                    }
                ),
                # Chart 1
                html.Div(
                    children=dcc.Graph(
                        figure=line_fig,
                        id='line_1'
                    ),
                    style={
                        'flex': '1',                    # Equal space for all charts
                        'margin': '10px',
                        'background-color': backgroundcolor,
                        'padding': '10px'
                    }
                ),
                # Chart 2
                html.Div(
                    children=dcc.Graph(
                        figure=new_bar,
                        id='bar_1'
                    ),
                    style={
                        'flex': '1',                    # Equal space for all charts
                        'margin': '10px',
                        'background-color': backgroundcolor,
                        'padding': '10px'
                    }
                ),
                # Explanation button and output
                html.Div(
                    children=[
                        html.Button(
                            "Explain Data",
                            id="explain_button",
                            n_clicks=0,
                            style={
                                'padding': '10px 20px',
                                'font-size': '16px',
                                'background-color': '#4CAF50',
                                'color': 'white',
                                'border': 'none',
                                'cursor': 'pointer',
                                'margin-top': '10px'
                                }
                                ),
                                html.Div(
                                    id="explanation_output",
                                    style={
                                        'margin-top': '20px',
                                        'padding': '10px',
                                        'font-size': '16px',
                                        'color': '#333333',
                                        'background-color': '#f9f9f9',
                                        'border': '1px solid #dddddd',
                                        'border-radius': '5px',
                                        'max-width': '600px',
                                        'text-align': 'center'
                                        }
                                        )
                                        ],
                                        style={
                                            'display': 'flex',
                                            'flexDirection': 'column',
                                            'alignItems': 'center',
                                            'justifyContent': 'center'
                                            }
                                            ),


            ],
            style={
                'display': 'flex',                      # Arrange charts in a row
                'flexDirection': 'row',                 # Horizontal alignment
                'alignItems': 'flex-start',             # Align charts to the top
                'justifyContent': 'space-around',       # Space out charts evenly
                'width': '100%'                         # Full width container
            }
        )
    ]
)

#===== call backs=========================================#

# country callback
@arms.callback(
        Output('country_dropdown','options'),
        Input('arms_dropdown','value')
)
def update_country_options(selected_roles):
    if selected_roles:
        filtered_data = geo[geo['Role'].isin(selected_roles)]
        countries = filtered_data['Country'].unique()
        return[{'label': country, 'value':country} for country in countries]
    all_countries = geo['Country'].unique()
    return [{'label':country, 'value':country} for country in all_countries ]


# Callback to update the map, line, and bar charts
@arms.callback(
    [Output('map_1', 'figure'), Output('line_1', 'figure'), Output('bar_1', 'figure')],
    [Input('arms_dropdown', 'value'), Input('country_dropdown','value')]
)
def update_charts(selected_roles,selected_countries):
    # If no selection, return the initial figures
    if not selected_roles and not selected_countries:
        return map_figure, line_fig, new_bar
    


# filtered data based on selected roles
    filtered_data_map = geo
    filtered_data_line = sup_rec_melt
    filtered_data_bar = ct_sort_melt

    

    # applying filter based on selected roles
    if selected_roles:
        filtered_data_map = filtered_data_map[filtered_data_map['Role'].isin(selected_roles)]
        filtered_data_line = filtered_data_line[filtered_data_line['Role'].isin(selected_roles)]
        filtered_data_bar = filtered_data_bar[filtered_data_bar['Role'].isin(selected_roles)]
    
    # applying filter based on selected countries
    if selected_countries:
        filtered_data_map = filtered_data_map[filtered_data_map['Country'].isin(selected_countries)]
        filtered_data_line = filtered_data_line[filtered_data_line['value'].isin(selected_countries)]
        filtered_data_bar = filtered_data_bar[filtered_data_bar['Country'].isin(selected_countries)]

    # If filtered data is empty, return empty figures
    if filtered_data_map.empty or filtered_data_line.empty or filtered_data_bar.empty:
        return go.Figure(), go.Figure(), go.Figure()
    



    # Create the scatter_geo map
    new_map = px.scatter_geo(
        filtered_data_map,
        locations='ISO_Code',
        projection='orthographic',
        color='Role',
        opacity=0.5,
        hover_name='Country',
        hover_data=['Order date', 'Numbers delivered','Predicted Numbers', 'Status', 'Recipient Affiliation', 'Armament category'],
        color_discrete_sequence=['#FF5733', '#33FF57']
    )
    new_map.update_geos(
        landcolor='lightgray',
        oceancolor='lightblue',
        coastlinecolor='darkblue',
        showland=True,
        showocean=True,
        showcoastlines=True
    )
    new_map.update_layout(
        title={'text': 'Global Arms Transfers: Supplier and Recipient Map', 'xanchor': 'center', 'x': 0.5},
        plot_bgcolor=backgroundcolor,
        paper_bgcolor=backgroundcolor,
        font_color=fontcolor,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            bgcolor=backgroundcolor,
            showframe=False,
            showcoastlines=True,
            coastlinecolor=fontcolor
        )
    )

    # Line chart with forecast integration
    new_line_fig = px.line(
        filtered_data_line,
        x='Delivery year',
        y='Numbers delivered',
        color='Role',
        color_discrete_sequence=['#ff0003', '#33FF57']
    )
    new_line_fig.add_scatter(
        x=filtered_data_line['Delivery year'],
        y=filtered_data_line['Predicted Numbers'],
        mode='lines',
        name='Forecast',
        line=dict(dash='dash', color='blue')
    )
    new_line_fig.update_layout(
        title={'text': 'Supply and Receipt of Armaments Over Time', 'xanchor': 'center', 'x': 0.5},
        plot_bgcolor=backgroundcolor,
        paper_bgcolor=backgroundcolor,
        font_color=fontcolor,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )

    # Bar chart
    # Create the original bar chart for actual data
    new_bar_fig = px.bar(
        ct_sort_melt,
        x='Armament category',
        y='Value',
        color = 'Type',
        barmode = 'group',
        color_discrete_map=color_discrete_map # Set color for the actual data
        )

    # Customize the layout
    new_bar_fig.update_layout(
        title={
            'text': 'Total Armaments Supplied & Received (Including Forecast)',
            'xanchor': 'center',
            'yanchor': 'middle',
            'x': 0.5
            },
    plot_bgcolor=backgroundcolor,
    paper_bgcolor=backgroundcolor,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    font_color=fontcolor
)
    new_bar_fig.update_traces(marker=dict(line=dict(width=0)))  # Remove outlines

    
    # Return the updated figures
    return new_map, new_line_fig, new_bar_fig

# summary callback

# Callback for the explanation button
@arms.callback(
    Output('explanation_output', 'children'),
    [Input('arms_dropdown', 'value'), 
     Input('country_dropdown', 'value')]
)
def explain_selection(selected_roles, selected_countries):
    # Map explanation: Summarizes countries and roles selected
    if selected_countries:
        map_explanation = f"The map shows data for {', '.join(selected_countries)} with roles as {', '.join(selected_roles)}."
    else:
        map_explanation = "The map shows data for the selected countries and roles."

    # Line chart explanation: Summarizes actual and predicted deliveries based on roles
    line_explanation = ""
    if selected_roles:
        # Filter the line chart data based on roles and countries
        filtered_data_line = sup_rec_melt[sup_rec_melt['Role'].isin(selected_roles)]

        # For each role, summarize predicted and actual deliveries
        for role in selected_roles:
            role_data = filtered_data_line[filtered_data_line['Role'] == role]
            actual_deliveries = role_data['Numbers delivered'].sum()
            predicted_deliveries = role_data['Predicted Numbers'].sum()
            line_explanation += f"The line chart shows actual and predicted deliveries for the role '{role}'. Actual deliveries: {actual_deliveries}, Predicted deliveries: {predicted_deliveries}. "

    # Bar chart explanation (add if needed, similar to line chart logic)
    bar_explanation = ""
    # Example: Summarizing armament categories and values for bar chart
    # bar_explanation += "Bar chart shows the total values for each armament category."

    # Combine all the explanations
    explanation = f"{map_explanation} {line_explanation} {bar_explanation}"
    
    return explanation





# Run the app
if __name__ == '__main__':
    arms.run_server(debug=True,port = 8052)
