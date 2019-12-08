# -*- coding: utf-8 -*-
import os
import dash

from visualization import create_layout, set_callbacks

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
app.title = "FDS2019"

server = app.server
app.layout = create_layout(app)
set_callbacks(app)

# Running server
if __name__ == "__main__":
    app.run_server(debug=True)
