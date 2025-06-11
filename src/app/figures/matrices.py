from math import ceil

from plotly import graph_objects as go
from plotly.subplots import make_subplots


def build_matrices_figure(matrices, t, desc, n_cols=5):
    # create figure for matrices to display
    ids = list(matrices.keys())
    n = len(matrices)
    n_rows = ceil(n / n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        vertical_spacing=0.05, horizontal_spacing=0.05)

    for i, ident in enumerate(ids):
        row = i // n_cols + 1
        col = i % n_cols + 1
        hm = go.Heatmap(
            name='/'.join(ident),
            x=desc['Name_Area'],
            y=desc['Name_Area'],
            z=(matrices[ident][t]),
            zmin=-1, zmax=1,
            coloraxis='coloraxis',
            showscale=False,
            hovertemplate='Row: %{y}'+
                          '<br>Column: %{x}<br>'+
                          'Corr: %{z:.2f}',
        )
        fig.add_trace(hm, row=row, col=col)

    # set up the layout
    fig.update_layout(coloraxis=dict(colorscale='RdBu_r', cmin=-1, cmax=1), showlegend=False)
    fig.update_layout(coloraxis_colorbar=dict(title=dict(text="Correlation", side='top')))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange='reversed')

    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.update_xaxes(constrain='domain', row=i, col=j)
            fig.update_yaxes(scaleanchor=f'x{(i-1)*n_cols+j}', scaleratio=1,
                             constrain='domain', row=i, col=j)

    return fig