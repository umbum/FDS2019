import base64
import io
import pathlib
import json
import pprint
import codecs

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from PIL import Image
from io import BytesIO
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
import plotly.graph_objs as go
import scipy.spatial.distance as spatial_distance

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

origin_datas = {
    # numerical dataset : corresponding original dataset
    "bs140513": pd.read_csv(DATA_PATH.joinpath("bs140513_origin.csv")),
}

# Import datasets here for running the Local version
TRANSACTION_DATASETS = ("bs140513")


with codecs.open(PATH.joinpath("intro.md"), "r", encoding="utf-8") as file:
    intro_md = file.read()

with codecs.open(PATH.joinpath("README.md"), "r", encoding="utf-8") as file:
    readme_md = file.read()


# Methods for creating components in the layout code
def Card(children, **kwargs):
    return html.Section(children, className="card-style")


def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f"div-{short}",
        style={"display": "inline-block"},
        children=[
            f"{name}:",
            dcc.RadioItems(
                id=f"radio-{short}",
                options=options,
                value=val,
                labelStyle={"display": "inline-block", "margin-right": "7px"},
                style={"display": "inline-block", "margin-left": "7px"},
            ),
        ],
    )


def create_layout(app):
    # Actual layout of the app
    return html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        [
                            html.H3(
                                "신용카드 이상 거래 클러스터링 및 시각화",
                                className="header_title",
                                id="app-title",
                            )
                        ],
                        className="nine columns header_title_container",
                    ),
                ],
            ),
            # Demo Description
            html.Div(
                className="row background",
                id="demo-explanation",
                style={"padding": "50px 45px"},
                children=[
                    html.Div(
                        id="description-text", children=dcc.Markdown(intro_md)
                    ),
                    html.Div(
                        html.Button(id="learn-more-button", children=["Learn More"])
                    ),
                ],
            ),
            # Body
            html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        children=[
                            Card(
                                [
                                    dcc.Dropdown(
                                        id="dropdown-dataset",
                                        searchable=False,
                                        clearable=False,
                                        options=[
                                            {
                                                "label": "BankSim",
                                                "value": "bs140513",
                                            },
                                            {
                                                "label": "PaySim",
                                                "value": "bs140513_autoencoded",
                                            }
                                        ],
                                        placeholder="Select a dataset",
                                        value="bs140513",
                                    ),
                                    NamedSlider(
                                        name="Number Of Iterations",
                                        short="iterations",
                                        min=250,
                                        max=1000,
                                        step=None,
                                        val=500,
                                        marks={
                                            i: str(i) for i in [250, 500, 750, 1000]
                                        },
                                    ),
                                    NamedSlider(
                                        name="Perplexity",
                                        short="perplexity",
                                        min=3,
                                        max=100,
                                        step=None,
                                        val=30,
                                        marks={i: str(i) for i in [3, 10, 30, 50, 100]},
                                    ),
                                    NamedSlider(
                                        name="Initial PCA Dimensions",
                                        short="pca-dimension",
                                        min=25,
                                        max=100,
                                        step=None,
                                        val=25,
                                        marks={i: str(i) for i in [25, 50, 100]},
                                    ),
                                    NamedSlider(
                                        name="Learning Rate",
                                        short="learning-rate",
                                        min=10,
                                        max=200,
                                        step=None,
                                        val=200,
                                        marks={i: str(i) for i in [10, 50, 100, 200]},
                                    )
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[
                            dcc.Graph(id="graph-3d-plot-tsne", style={"height": "98vh"})
                        ],
                    ),
                    html.Div(
                        className="three columns",
                        id="euclidean-distance",
                        children=[
                            Card(
                                style={"padding": "5px"},
                                children=[
                                    html.Div(
                                        id="div-plot-click-message",
                                        style={
                                            "text-align": "center",
                                            "margin-bottom": "7px",
                                            "font-weight": "bold",
                                        },
                                    ),
                                    html.Div(id="div-plot-click-image"),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )


def set_callbacks(app):
    def generate_figure(groups, layout):
        data = []
        types = {
            "TN" : {
                "symbol":"cross",
                "opacity":0.5,
                "size": 4,
            },
            "TP" : {
                "symbol":"square",
                "opacity":0.9,
                "size": 4,
            },
            "FN" : {
                "symbol":"diamond",
                "opacity":0.9,
                "size": 3,
            },
            "FP" : {
                "symbol":"circle",
                "opacity":0.8,
                "size": 4,
            },
        }

        for group_key, val in groups:
            label, cluster = group_key
            if label == 0 and cluster == 0:
                name = "TN"
            elif label == 0 and cluster == 1:
                name = "FP"
            elif label == 1 and cluster == 0:
                name = "FN"
            else:
                name = "TP"

            scatter = go.Scatter3d(
                name=name,
                x=val["x"],
                y=val["y"],
                z=val["z"],
                text=[label for _ in range(val["x"].shape[0])],
                textposition="top center",
                mode="markers",
                marker=dict(symbol=types[name]["symbol"],
                    size=types[name]["size"], 
                    opacity=types[name]["opacity"]),
            )
            data.append(scatter)

        figure = go.Figure(data=data, layout=layout)

        return figure

    # Callback function for the learn-more button
    @app.callback(
        [
            Output("description-text", "children"),
            Output("learn-more-button", "children"),
        ],
        [Input("learn-more-button", "n_clicks")],
    )
    def learn_more(n_clicks):
        # If clicked odd times, the instructions will show; else (even times), only the header will show
        if n_clicks == None:
            n_clicks = 0
        if (n_clicks % 2) == 1:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(readme_md)],
                ),
                "Close",
            )
        else:
            n_clicks += 1
            return (
                html.Div(
                    style={"padding-right": "15%"},
                    children=[dcc.Markdown(intro_md)],
                ),
                "Learn More",
            )

    @app.callback(
        Output("graph-3d-plot-tsne", "figure"),
        [
            Input("dropdown-dataset", "value"),
            Input("slider-iterations", "value"),
            Input("slider-perplexity", "value"),
            Input("slider-pca-dimension", "value"),
            Input("slider-learning-rate", "value"),
        ],
    )
    def display_3d_scatter_plot(dataset, iterations, perplexity, 
                                pca_dim, learning_rate):
        if dataset:
            path = f"embeddings/{dataset}/iterations_{iterations}/perplexity_{perplexity}/pca_{pca_dim}/learning_rate_{learning_rate}"

            try:

                data_url = [
                    "embeddings",
                    str(dataset),
                    "iterations_" + str(iterations),
                    "perplexity_" + str(perplexity),
                    "pca_" + str(pca_dim),
                    "learning_rate_" + str(learning_rate),
                    "data.csv",
                ]
                full_path = PATH.joinpath(*data_url)
                embedding_df = pd.read_csv(
                    full_path, index_col=0, encoding="ISO-8859-1"
                )

            except FileNotFoundError as error:
                print(
                    error,
                    "\nThe dataset was not found. Please generate it using generate_embeddings.py",
                )
                return go.Figure()

            # Plot layout
            axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

            layout = go.Layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
            )

            if dataset in TRANSACTION_DATASETS:
                embedding_df["label"] = embedding_df.index

                groups = embedding_df.groupby(["label", "cluster"])
                figure = generate_figure(groups, layout)

            else:
                figure = go.Figure()

            return figure

    @app.callback(
        Output("div-plot-click-image", "children"),
        [Input("graph-3d-plot-tsne", "clickData"),
         Input("dropdown-dataset", "value")],
        [State("slider-iterations", "value"),
         State("slider-perplexity", "value"),
         State("slider-pca-dimension", "value"),
         State("slider-learning-rate", "value")],
    )
    def display_click_point(clickData, dataset, iterations, 
                            perplexity, pca_dim, learning_rate):
        if not clickData:
            return None

        try:
            data_url = [
                "embeddings",
                str(dataset),
                "iterations_" + str(iterations),
                "perplexity_" + str(perplexity),
                "pca_" + str(pca_dim),
                "learning_rate_" + str(learning_rate),
                "data.csv",
            ]
            full_path = PATH.joinpath(*data_url)
            embedding_df = pd.read_csv(full_path, encoding="ISO-8859-1")

        except FileNotFoundError as error:
            print(
                error,
                "\nThe dataset was not found. Please generate it using generate_embeddings.py",
            )
            return

        # Convert the point clicked into float64 numpy array
        click_point_np = np.array(
            [clickData["points"][0][i] for i in ["x", "y", "z"]]
        ).astype(np.float64)
        # Create a boolean mask of the point clicked, truth value exists at only one row
        bool_mask_click = (
            embedding_df.loc[:, "x":"z"].eq(click_point_np).all(axis=1)
        )
        # Retrieve the index of the point clicked, given it is present in the set
        if bool_mask_click.any():
            clicked_idx = embedding_df[bool_mask_click].index[0]

            # Retrieve the data corresponding to the index (Dimension reduction 이전의 원래 vector)
            origin_vector = origin_datas[dataset].iloc[clicked_idx]
            
            return html.Pre(children=pprint.pformat(origin_vector.to_dict()))

    @app.callback(
        Output("div-plot-click-message", "children"),
        [Input("graph-3d-plot-tsne", "clickData"), Input("dropdown-dataset", "value")],
    )
    def display_click_message(clickData, dataset):
        # Displays message shown when a point in the graph is clicked, depending whether it's an image or word
        if dataset in TRANSACTION_DATASETS:
            if clickData:
                return "Transaction Selected"
            else:
                return "Click a data point on the scatter plot to display its corresponding information."

