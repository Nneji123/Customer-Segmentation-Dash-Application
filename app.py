import pandas as pd
import plotly.express as px
import wqet_grader
from dash import Input, Output, dcc, html
from IPython.display import VimeoVideo
from jupyter_dash import JupyterDash
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

JupyterDash.infer_jupyter_proxy_config()


def wrangle(filepath):
    """Read SCF data file into ``DataFrame``.

    Returns only credit fearful households whose net worth is less than $2 million.

    Parameters
    ----------
    filepath : str
        Location of CSV file.
    """
    # Load data
    df = pd.read_csv(filepath)
    # Create mask
    mask = (df["TURNFEAR"] == 1) & (df["NETWORTH"] < 2e6)
    # Subset DataFrame
    df = df[mask]

    return df


df = wrangle("scfp2019.zip")

app = JupyterDash(__name__)


def get_high_var_features(trimmed=True, return_feat_names=True):

    """Returns the five highest-variance features of ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    return_feat_names : bool, default=False
        If ``True``, returns feature names as a ``list``. If ``False``
        returns ``Series``, where index is feature names and values are
        variances.
    """
    # Calculate Variance
    if trimmed:
        top_five_features = df.apply(trimmed_var).sort_values().tail(5)
    else:
        top_five_features = df.var().sort_values().tail(5)

    # Extract names
    if return_feat_names:
        top_five_features = top_five_features.index.tolist()
    return top_five_features


@app.callback(Output("bar-chart", "figure"), Input("trim-button", "value"))
def serve_bar_chart(trimmed=True):

    """Returns a horizontal bar chart of five highest-variance features.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.
    """
    # Get features
    top_five_features = get_high_var_features(trimmed=trimmed, return_feat_names=False)
    # Build bar chart
    fig = px.bar(x=top_five_features, y=top_five_features.index, orientation="h")
    fig.update_layout(xaxis_title="Variance", yaxis_title="Feature")

    return fig


def get_model_metrics(trimmed=True, k=2, return_metrics=False):

    """Build ``KMeans`` model based on five highest-variance features in ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.

    return_metrics : bool, default=False
        If ``False`` returns ``KMeans`` model. If ``True`` returns ``dict``
        with inertia and silhouette score.

    """
    # Get high var features
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    # Create feature matrix
    X = df[features]
    # Build model
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
    model.fit(X)

    if return_metrics:
        # Calculate Inertia
        i = model.named_steps["kmeans"].inertia_
        # Calculate Silhouette score
        ss = silhouette_score(X, model.named_steps["kmeans"].labels_)
        # Put results into dictionary
        metrics = {"inertia": round(i), "silhouette": round(ss, 3)}
        # return dictionary to users
        return metrics

    return model


@app.callback(
    Output("metrics", "children"),
    Input("trim-button", "value"),
    Input("k-slider", "value"),
)
def serve_metrics(trimmed=True, k=2):

    """Returns list of ``H3`` elements containing inertia and silhouette score
    for ``KMeans`` model.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    # Get metrics
    metrics = get_model_metrics(trimmed=trimmed, k=k, return_metrics=True)

    # Add metrics to HTML elements
    text = [
        html.H3(f"Inertia: {metrics['inertia']}"),
        html.H3(f"Silhouette Score: {metrics['silhouette']}"),
    ]

    return text


def get_pca_labels(trimmed=True, k=2):

    """
    ``KMeans`` labels.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    # Create feature matrix
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]

    # Build Transformer
    transformer = PCA(n_components=2, random_state=42)

    # Transform Data
    X_t = transformer.fit_transform(X)
    X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])

    # Add labels
    model = get_model_metrics(trimmed=trimmed, k=k, return_metrics=False)
    X_pca["labels"] = model.named_steps["kmeans"].labels_.astype(str)
    X_pca.sort_values("labels", inplace=True)

    return X_pca


@app.callback(
    Output("pca-scatter", "figure"),
    Input("trim-button", "value"),
    Input("k-slider", "value"),
)
def serve_scatter_plot(trimmed=True, k=2):

    """Build 2D scatter plot of ``df`` with ``KMeans`` labels.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.
    """
    fig = px.scatter(
        data_frame=get_pca_labels(trimmed=trimmed, k=k),
        x="PC1",
        y="PC2",
        color="labels",
        title="PCA Representation of Clusters",
    )
    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return fig


app.layout = html.Div(
    [
        # Application title
        html.H1("Survey of Consumer Finances"),
        # Bar chart element
        html.H2("High Variance Features"),
        # Bar chart graph
        dcc.Graph(figure=serve_bar_chart(), id="bar-chart"),
        dcc.RadioItems(
            options=[
                {"label": "trimmed", "value": True},
                {"label": "not trimmed", "value": False},
            ],
            value=True,
            id="trim-button",
        ),
        html.H2("K-means Clustering"),
        html.H3("Number of Clusters (k)"),
        dcc.Slider(min=2, max=12, step=1, value=2, id="k-slider"),
        html.Div(id="metrics"),
        # PCA scatter plot
        dcc.Graph(id="pca-scatter"),
    ]
)

app.run_server(host="0.0.0.0", mode="external")
