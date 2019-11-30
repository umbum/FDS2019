import plotly.express as px

iris = px.data.iris()
fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_width',
              color='petal_length', opacity=0.7, symbol='species',
              size='petal_length', size_max=22)

fig.update_layout(
    autosize=False,
    width=1400,
    height=700,
    margin=dict(l=0, r=0, b=0, t=0)
)

fig.show()