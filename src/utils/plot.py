import math

import matplotlib
from matplotlib import pyplot as plt
from plotly.validators.scatter.marker import SymbolValidator
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np
from plotly.subplots import make_subplots
from plotly.tools import DEFAULT_PLOTLY_COLORS

from src.models.compare.winners import kruskal_significance
from src.utils.files import create_dir, get_new_file_path
import plotly.express as px

pio.renderers.default = "browser"

plotly_colors = [
    '#1f77b4',  # muted blue
    '#2ca02c',  # cooked asparagus green
    '#ff7f0e',  # safety orange
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf',  # blue-teal,
    'rgba(1, 1, 1, 1)'  # black
]

template = 'plotly_white'

raw_symbols = SymbolValidator().values
namestems = []
namevariants = []
symbols = []
for i in range(0, len(raw_symbols), 3):
    name = raw_symbols[i + 2]
    symbols.append(raw_symbols[i])
    namestems.append(name.replace("-open", "").replace("-dot", ""))
    namevariants.append(name[len(namestems[-1]):])

marker_names = {}
for name in namestems:
    if name not in marker_names:
        marker_names[name] = 1
marker_names = list(marker_names.keys())

color_start = 4
color_sequences = [px.colors.sequential.Blues[color_start:],
                   px.colors.sequential.Reds[color_start:],
                   px.colors.sequential.Greens[color_start:],
                   px.colors.sequential.Purples[color_start:],
                   px.colors.sequential.Oranges[color_start:]]


def confidence_ellipse(x, y, n_std=1.96, size=100):
    """
    Get the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    size : int
        Number of points defining the ellipse

    Returns
    -------
    String containing an SVG path for the ellipse

    References (H/T)
    ----------------
    https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
    https://community.plotly.com/t/arc-shape-with-path/7205/5
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack([ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)])

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean], (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
                                [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    scale_matrix = np.array([[x_scale, 0],
                             [0, y_scale]])
    ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(scale_matrix) + translation_matrix

    path = f'M {ellipse_coords[0, 0]}, {ellipse_coords[0, 1]}'
    for k in range(1, len(ellipse_coords)):
        path += f'L{ellipse_coords[k, 0]}, {ellipse_coords[k, 1]}'
    path += ' Z'
    return path


def plotly_save(fig, file_path, size, save_png=False, use_date_suffix=False):
    print("Saving image:")
    create_dir(file_path)
    image_path = get_new_file_path(file_path, '.png', use_date_suffix)
    html_path = get_new_file_path(file_path, '.html', use_date_suffix)
    if size is None:
        size = (1980, 1080)

    if save_png:
        print(image_path)
        fig.write_image(image_path, width=size[0], height=size[1], engine='orca')

    print(html_path)
    fig.write_html(html_path)


def plot_2D_predictor_corrector(points,
                                predictors,
                                correctors,
                                descent=None,
                                pareto=None,
                                scale=1,
                                arrow_scale=0.4,
                                point_name='x',
                                markersize=6,
                                line_width=1,
                                plot_arrows=True,
                                plot_points=True,
                                save=False,
                                save_png=False,
                                file_path=None,
                                size=(1980, 1080),
                                return_fig=False,
                                title=None,
                                plot_title=True,
                                overlayed=False,  # when this fig will be overlayed on top
                                pareto_marker_mode='lines'):
    c_points, c_uv, c_xy = get_corrector_arrows(correctors)
    p_points, p_uv, p_xy = get_predictor_arrows(predictors, points)
    # c_points, c_uv, c_xy, p_points, p_uv, p_xy = get_arrows(correctors, points, predictors)

    if descent is not None:
        d_points, d_uv, d_xy = get_corrector_arrows(descent)

    fig = make_subplots(rows=1, cols=1)

    if pareto is not None and not overlayed:
        pareto_plot = pareto.to_numpy() if isinstance(pareto, pd.DataFrame) else pareto
        fig.add_trace(go.Scatter(x=pareto_plot[:, 0],
                                 y=pareto_plot[:, 1],
                                 marker=dict(color=plotly_colors[0]),
                                 mode=pareto_marker_mode,
                                 showlegend=not overlayed,
                                 name='pareto set' if point_name == 'x' else 'pareto front'))

    if plot_arrows:
        p_quiver_fig = ff.create_quiver(p_xy[:, 0], p_xy[:, 1], p_uv[:, 0], p_uv[:, 1],
                                        scale=scale,
                                        arrow_scale=arrow_scale,
                                        showlegend=not overlayed,
                                        name='predictor',
                                        marker=dict(color=plotly_colors[3]),
                                        line_width=line_width)

        fig.add_traces(data=p_quiver_fig.data)

        if len(c_uv) > 0:
            c_quiver_fig = ff.create_quiver(c_xy[:, 0], c_xy[:, 1], c_uv[:, 0], c_uv[:, 1],
                                            scale=scale,
                                            arrow_scale=arrow_scale,
                                            showlegend=not overlayed,
                                            name='corrector',
                                            marker=dict(color=plotly_colors[1]),
                                            line_width=line_width)

            fig.add_traces(data=c_quiver_fig.data)

        if descent is not None and len(d_uv) > 0:
            d_quiver_fig = ff.create_quiver(d_xy[:, 0], d_xy[:, 1], d_uv[:, 0], d_uv[:, 1],
                                            scale=scale,
                                            arrow_scale=arrow_scale,
                                            showlegend=not overlayed,
                                            name='descent',
                                            marker=dict(color=plotly_colors[9]),
                                            line_width=line_width)

            fig.add_traces(data=d_quiver_fig.data)

    if plot_points:
        fig.add_trace(go.Scatter(x=p_points[:, 0],
                                 y=p_points[:, 1],
                                 mode='markers',
                                 showlegend=not overlayed and not plot_arrows,
                                 marker=dict(size=markersize,
                                             color=plotly_colors[3]),
                                 name='predictor'))

        if c_points.shape[0] > 0:
            fig.add_trace(go.Scatter(x=c_points[:, 0],
                                     y=c_points[:, 1],
                                     mode='markers',
                                     showlegend=not overlayed and not plot_arrows,
                                     marker=dict(size=markersize,
                                                 color=plotly_colors[1]),
                                     name='corrector'))

    fig.add_trace(go.Scatter(x=points[:, 0],
                             y=points[:, 1],
                             mode='markers',
                             showlegend=not overlayed,
                             marker=dict(size=markersize * 1.5,
                                         color='rgba(1, 1, 1, 1)'),
                             name=point_name))

    if not overlayed and descent is not None and len(d_uv) > 0:
        fig.add_trace(go.Scatter(x=d_xy[:1, 0],
                                 y=d_xy[:1, 1],
                                 mode='markers',
                                 marker_symbol='x',
                                 marker=dict(size=markersize * 3,
                                             color='rgba(1, 1, 1, 1)'),
                                 name='initial'))

    if return_fig:
        return fig

    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    if plot_title:
        fig.update_layout(title=title)

    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def get_predictor_arrows(predictors, points):
    p_xy, p_uv, p_points = [], [], []
    # predictors
    for preds, point in zip(predictors, points):
        # for corrs, preds, point in zip(correctors, predictors, points[:-1]):
        for p in preds:
            p_xy.append(point)
            p_uv.append(p - point)
            p_points.append(p)

    p_xy, p_uv, p_points = np.array(p_xy), np.array(p_uv), np.array(p_points)
    return p_points, p_uv, p_xy


def get_corrector_arrows(correctors):
    c_uv, c_xy, c_points = [], [], []
    for corrs in correctors:
        for i, c in enumerate(corrs):
            if i > 0:
                c_points.append(c)
                c_xy.append(corrs[i - 1])
                c_uv.append(corrs[i] - corrs[i - 1])

    c_points, c_uv, c_xy = np.array(c_points), np.array(c_uv), np.array(c_xy)
    return c_points, c_uv, c_xy


def get_arrows(correctors, points, predictors):
    p_xy, p_uv, p_points, c_uv, c_xy, c_points = [], [], [], [], [], []
    # predictors
    for corrs, preds, point in zip(correctors, predictors, points):
        # for corrs, preds, point in zip(correctors, predictors, points[:-1]):
        for p in preds:
            p_xy.append(point)
            p_uv.append(p - point)
            p_points.append(p)

        for i, c in enumerate(corrs):
            if i > 0:
                c_points.append(c)
                c_xy.append(corrs[i - 1])
                c_uv.append(corrs[i] - corrs[i - 1])
    p_xy, p_uv, p_points = np.array(p_xy), np.array(p_uv), np.array(p_points)
    c_points, c_uv, c_xy = np.array(c_points), np.array(c_uv), np.array(c_xy)
    return c_points, c_uv, c_xy, p_points, p_uv, p_xy


def plot_2D_points_traces_total(points_traces,
                                names=None,
                                markersizes=12,
                                color_ixs=None,
                                modes=None,
                                marker_symbols=None,
                                outlines=None,
                                save=False,
                                save_png=False,
                                file_path=None,
                                title=None,
                                size=(1980, 1080),
                                axes_labels=None,
                                show_legends=None,
                                label_scale=1,
                                ):
    fig = make_subplots(rows=1, cols=2,
                        shared_xaxes=True,
                        subplot_titles=('Pareto Front', 'Total'))

    for i, points in enumerate(points_traces):
        fig.add_trace(go.Scatter(x=points[:, 0],
                                 y=points[:, 1],
                                 mode='markers' if modes is None else modes[i],
                                 marker_symbol=None if marker_symbols is None else marker_symbols[i],
                                 marker=dict(size=markersizes if isinstance(markersizes, int) else markersizes[i],
                                             color=None if color_ixs is None else plotly_colors[
                                                 color_ixs[i] % len(plotly_colors)],
                                             line=None if outlines is None else dict(width=2,
                                                                                     color='black') if outlines[
                                                 i] else None,
                                             ),
                                 showlegend=show_legends[i] if show_legends is not None else True,
                                 name=None if names is None else names[i]), row=1, col=1)

        fig.add_trace(go.Scatter(x=points[:, 0],
                                 y=np.sum(points, axis=1),
                                 mode='markers' if modes is None else modes[i],
                                 marker_symbol=None if marker_symbols is None else marker_symbols[i],
                                 marker=dict(size=markersizes if isinstance(markersizes, int) else markersizes[i],
                                             color=None if color_ixs is None else plotly_colors[
                                                 color_ixs[i] % len(plotly_colors)],
                                             line=None if outlines is None else dict(width=2,
                                                                                     color='black') if outlines[
                                                 i] else None,
                                             ),
                                 showlegend=False,
                                 name=None if names is None else names[i] + '_total'), row=1, col=2)

    if axes_labels is not None:
        for col in [1, 2]:
            fig.update_xaxes(title_text=axes_labels[0], row=1, col=col)
            fig.update_yaxes(title_text=axes_labels[1] if col == 1 else 'Total', row=1, col=col)
    fig.update_layout(title=title,
                      template=template,
                      legend=dict(font=dict(size=18 * label_scale)),
                      font_color="black")
    fig.update_annotations(font_size=14 * label_scale)
    fig.update_xaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))
    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def plot_2D_points_traces(points_traces,
                          names=None,
                          markersizes=12,
                          color_ixs=None,
                          modes=None,
                          save=False,
                          save_png=False,
                          file_path=None,
                          title=None,
                          size=(1980, 1080),
                          axes_labels=None,
                          ):
    fig = make_subplots(rows=1, cols=1)

    for i, points in enumerate(points_traces):
        fig.add_trace(go.Scatter(x=points[:, 0],
                                 y=points[:, 1],
                                 mode='markers' if modes is None else modes[i],
                                 marker=dict(size=markersizes if isinstance(markersizes, int) else markersizes[i],
                                             color=None if color_ixs is None else plotly_colors[color_ixs[i]]),
                                 name=None if names is None else names[i]))

    if axes_labels is not None:
        fig.update_xaxes(title_text=axes_labels[0])
        fig.update_yaxes(title_text=axes_labels[1])

    fig.update_layout(title=title)
    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def plot_2D_points_vectors(points,
                           vectors=None,
                           pareto=None,
                           scale=0.5,
                           arrow_scale=0.4,
                           point_name='x',
                           vector_name='v',
                           markersize=12,
                           save=False,
                           save_png=False,
                           file_path=None,
                           title=None,
                           size=(1980, 1080),
                           ):
    if vectors is not None and points.shape[0] > vectors.shape[0]:
        x, y = points[:-1, 0], points[:-1, 1]
    else:
        x, y = points[:, 0], points[:, 1]

    fig = make_subplots(rows=1, cols=1)

    if vectors is not None:
        u, v = vectors[:, 0], vectors[:, 1]
        quiver_fig = ff.create_quiver(x, y, u, v,
                                      scale=scale,
                                      arrow_scale=arrow_scale,
                                      name=vector_name,
                                      line_width=2)

        fig.add_traces(data=quiver_fig.data)

    if pareto is not None:
        pareto_plot = pareto.to_numpy() if isinstance(pareto, pd.DataFrame) else pareto
        fig.add_trace(go.Scatter(x=pareto_plot[:, 0],
                                 y=pareto_plot[:, 1],
                                 mode='lines',
                                 name='pareto front'))

    fig.add_trace(go.Scatter(x=x,
                             y=y,
                             mode='markers',
                             marker=dict(size=markersize),
                             name=point_name))

    fig.add_trace(go.Scatter(x=x[:1],
                             y=y[:1],
                             mode='markers',
                             marker_symbol='x',
                             marker=dict(size=markersize,
                                         color='rgba(1, 1, 1, 1)'),
                             name='initial'))

    fig.update_layout(title=title)
    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def plot_bidir_2D_points_vectors(results,
                                 pareto=None,
                                 descent=None,
                                 arrow_scale=0.4,
                                 markersize=6,
                                 save=False,
                                 save_png=False,
                                 file_path=None,
                                 size=(1980, 1080),
                                 plot_arrows=True,
                                 plot_points=True,
                                 plot_ps=True,
                                 pareto_marker_mode='lines',
                                 return_fig=False,
                                 titles=None,
                                 line_width=2,
                                 plot_title=True,
                                 label_scale=1,
                                 ):
    results = [results] if not isinstance(results, list) else results
    xs_figs, fxs_figs = [], []
    if titles is None:
        titles = ['Continuation method: predictors and correctors in decision space',
                  'Continuation method: predictors and correctors in objective space']
    for i, res in enumerate(results):
        if plot_ps:
            x_fig = plot_2D_predictor_corrector(points=res['X'],
                                                predictors=res['X_p'],
                                                correctors=res['X_c'],
                                                pareto=pareto['ps'] if pareto is not None else None,
                                                descent=descent[i]['X'] if descent is not None else None,
                                                title=titles[i],
                                                arrow_scale=arrow_scale,
                                                point_name='x',
                                                markersize=markersize,
                                                line_width=line_width,
                                                plot_arrows=plot_arrows,
                                                plot_points=plot_points,
                                                return_fig=True,
                                                overlayed=i > 0,
                                                save=False,
                                                plot_title=plot_title,
                                                pareto_marker_mode=pareto_marker_mode
                                                )

        fx_fix = plot_2D_predictor_corrector(points=res['F'],
                                             predictors=res['F_p'],
                                             correctors=res['F_c'],
                                             pareto=pareto['pf'] if pareto is not None else None,
                                             descent=descent[i]['F'] if descent is not None else None,
                                             arrow_scale=arrow_scale,
                                             point_name='F(x)',
                                             markersize=markersize,
                                             line_width=line_width,
                                             plot_arrows=plot_arrows,
                                             plot_points=plot_points,
                                             return_fig=True,
                                             overlayed=i > 0,
                                             save=False,
                                             plot_title=plot_title,
                                             pareto_marker_mode=pareto_marker_mode
                                             )
        fxs_figs.append(fx_fix)
        if plot_ps:
            xs_figs.append(x_fig)

    if plot_ps:
        fig = make_subplots(rows=1, cols=1)
        [fig.add_traces(data=f.data) for f in xs_figs]
        fig.update_layout(template=template, legend=dict(font=dict(size=18 * label_scale)), font_color="black")
        fig.update_xaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))
        fig.update_yaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))
        fig.show()

        if file_path is not None and save is True:
            plotly_save(fig, file_path + '_ds', size, save_png=save_png)

    fig = make_subplots(rows=1, cols=1)
    [fig.add_traces(data=f.data) for f in fxs_figs]
    fig.update_layout(template=template, legend=dict(font=dict(size=18 * label_scale)), font_color="black")
    fig.update_xaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    if plot_title:
        fig.update_layout(title=titles[1])
    if return_fig:
        return fig
    else:
        fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path + '_os', size, save_png=save_png)


def plot_boxes_3d(boxes_edges,
                  return_fig=False):
    fig = make_subplots(rows=1, cols=1)

    opacities = np.exp2(np.arange(0, len(boxes_edges)))
    opacities /= max(opacities)
    # opacities = np.arange(0, 1, 1 / len(boxes_edges)) + 1 / len(boxes_edges)
    for l, level in enumerate(boxes_edges):
        for box in level:
            for edge in box:
                fig.add_trace(go.Scatter3d(x=[e[0] for e in edge],
                                           y=[e[1] for e in edge],
                                           z=[e[2] for e in edge],
                                           showlegend=False,
                                           opacity=opacities[l],
                                           line=dict(color=plotly_colors[4]),
                                           mode='lines'))

    if return_fig:
        return fig
    else:
        fig.show()


def plot_boxes_2d(boxes_edges,
                  return_fig=False):
    fig = make_subplots(rows=1, cols=1)

    opacities = np.exp2(np.arange(0, len(boxes_edges)))
    opacities /= max(opacities)

    for l, level in enumerate(boxes_edges):
        for box in level:
            for edge in box:
                fig.add_trace(go.Scatter(x=[e[0] for e in edge],
                                         y=[e[1] for e in edge],
                                         showlegend=False,
                                         opacity=opacities[l],
                                         line=dict(color=plotly_colors[4]),
                                         mode='lines'))

    if return_fig:
        return fig
    else:
        fig.show()


def plot_points_centers_2d(points,
                           centers=None,
                           best=None,
                           return_fig=False,
                           color_ix=[1, 4],
                           points_name='fx',
                           centers_name='box center',
                           markersize=8):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=points[:, 0],
                             y=points[:, 1],
                             showlegend=True,
                             marker=dict(size=markersize,
                                         color=plotly_colors[color_ix[0]]),
                             mode='markers',
                             name=points_name))

    if centers is not None:
        fig.add_trace(go.Scatter(x=centers[:, 0],
                                 y=centers[:, 1],
                                 showlegend=True,
                                 marker=dict(size=markersize,
                                             color=plotly_colors[color_ix[1]]),
                                 marker_symbol='cross',
                                 mode='markers',
                                 name=centers_name))

    if best is not None:
        fig.add_trace(go.Scatter(x=points[best, 0],
                                 y=points[best, 1],
                                 showlegend=True,
                                 marker=dict(size=markersize,
                                             color='rgba(1, 1, 1, 1)'),
                                 mode='markers',
                                 name='best'))

    if return_fig:
        return fig
    else:
        fig.show()


def plot_points_4d(points,
                   secondary=None,
                   mask=None,
                   return_fig=False,
                   color_ix=[1, 4],
                   points_name='fx',
                   secondary_name='box center',
                   secondary_marker_symbol='cross',
                   markersize=8,
                   only_best=False,
                   title=None):
    d4 = points.shape[1] > 3
    fig = make_subplots(rows=1, cols=1)
    if not only_best:
        fig.add_trace(go.Scatter3d(x=points[:, 0],
                                   y=points[:, 1],
                                   z=points[:, 2],
                                   showlegend=True,
                                   marker=dict(size=markersize / 1.5,
                                               color=points[:, 3] if d4 else plotly_colors[color_ix[0]]),
                                   opacity=0.7 if secondary is not None and not only_best else 1,
                                   mode='markers',
                                   name=points_name))

    if secondary is not None:
        fig.add_trace(go.Scatter3d(x=secondary[:, 0],
                                   y=secondary[:, 1],
                                   z=secondary[:, 2],
                                   showlegend=True,
                                   marker=dict(size=markersize,
                                               color=plotly_colors[color_ix[1]]),
                                   marker_symbol=secondary_marker_symbol,
                                   mode='markers',
                                   name=secondary_name))

    if mask is not None:
        fig.add_trace(go.Scatter3d(x=points[mask, 0],
                                   y=points[mask, 1],
                                   z=points[mask, 2],
                                   showlegend=True,
                                   marker=dict(size=markersize,
                                               color='rgba(1, 1, 1, 1)'),
                                   mode='markers',
                                   name='best'))

    if return_fig:
        return fig
    else:
        fig.update_layout(title=title)
        fig.show()


def plot_traces(data,
                file_path=None,
                save=False,
                save_png=False,
                size=(1980, 1080)):
    fig = make_subplots()
    for d in data:
        fig.add_traces(d)

    fig.show()
    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def plot_bars_from_results(key, results, labels, x_labels, return_fig=False):
    fig = go.Figure()
    for lbl, res in zip(labels, results):
        fig.add_trace(go.Bar(name='{}: {}'.format(lbl, key),
                             x=x_labels,
                             y=np.mean(res[key], axis=0),
                             error_y=dict(type='data', array=np.std(res[key], axis=0))
                             ))
    fig.update_layout(barmode='group')

    if return_fig:
        return fig
    fig.show()


def plot_classification_results(results, labels, keys, x_labels):
    figs = []
    for key in keys:
        figs.append(plot_bars_from_results(key, results, labels, x_labels, return_fig=True).data)

    fig = make_subplots(rows=1, cols=3, subplot_titles=keys)
    for col in range(3):
        for i in [0, 1]:
            fig.append_trace(figs[col][i], 1, col + 1)

    fig.show()


def plot_metrics_traces(traces, subtitles=None, x_labels=None):
    fig = make_subplots(rows=1, cols=traces[0].shape[0], subplot_titles=subtitles)
    for i, trace in enumerate(traces):
        for ind in trace:
            fig.append_trace(go.Box(x=x_labels,
                                    y=ind,
                                    showlegend=False),
                             row=1,
                             col=i + 1)

    fig.show()


def plot_pfs(Fs,
             fx_inis,
             names,
             save=False,
             file_path=None,
             size=(1980, 1080),
             save_png=False,
             axes_labels=None):
    n_cols = math.ceil(len(Fs) ** 0.5)
    n_rows = math.ceil(len(Fs) / n_cols)

    fig = make_subplots(rows=n_rows,
                        cols=n_cols,
                        shared_xaxes=True,
                        shared_yaxes=True,
                        subplot_titles=names)

    for i, (f, ini) in enumerate(zip(Fs, fx_inis)):
        fig.add_trace(go.Scatter(
            mode='markers',
            name=names[i],
            marker=dict(size=5, color=plotly_colors[i % len(plotly_colors)]),
            x=f[:, 0], y=f[:, 1],
        ), row=(i // n_cols) + 1, col=(i % n_cols) + 1)

        fig.add_trace(go.Scatter(
            mode='markers',
            marker_symbol='hexagram',
            marker=dict(size=15, color=plotly_colors[i % len(plotly_colors)], line=dict(width=1, color='black')),
            x=ini[:, 0], y=ini[:, 1],
            showlegend=False,
        ), row=(i // n_cols) + 1, col=(i % n_cols) + 1)

    if axes_labels is not None:
        fig.update_xaxes(title_text=axes_labels[0])
        fig.update_yaxes(title_text=axes_labels[1])

    fig.update_xaxes(matches='x', scaleratio=1, showticklabels=True)
    fig.update_yaxes(matches='x', scaleratio=1, showticklabels=True)

    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def bar_plots_with_errors(plot_dict_values, secondary_y=False, title=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    labels = list(plot_dict_values.keys())
    labels.remove('lbls')

    for i, key in enumerate(labels):
        values = plot_dict_values[key]
        fig.add_trace(go.Bar(
            name=key,
            marker=dict(color=plotly_colors[i]),
            offsetgroup=i + 1 if secondary_y else None,
            x=plot_dict_values['lbls'], y=values['mean'],
            error_y=dict(type='data', array=values['std']) if 'std' in values else None,
        ),
            secondary_y=i == 1 if secondary_y else False, )

    fig.update_layout(barmode='group', title=title)

    if secondary_y:
        for i in range(len(labels)):
            fig.update_yaxes(dict(color=plotly_colors[i]), secondary_y=i == 1)
            fig.update_yaxes(title=labels[i], secondary_y=i == 1)

    fig.show()


def bar_plot_3axes_with_errors(plot_dict_values, title=None):
    labels = list(plot_dict_values.keys())
    labels.remove('lbls')

    fig = go.Figure()

    color_ixs = [0, 1, 3]
    for i, (ix, key) in enumerate(zip(color_ixs, labels)):
        values = plot_dict_values[key]
        fig.add_trace(go.Bar(
            x=plot_dict_values['lbls'],
            y=values['mean'],
            name=key,
            marker=dict(color=plotly_colors[ix]),
            error_y=dict(type='data', array=values['std']) if 'std' in values else None,
            yaxis='y' + str(i + 1),
            offsetgroup=i + 1
        ))

    i = 0

    fig.update_layout(
        xaxis=dict(
            domain=[0.1, 0.9],
        ),
        yaxis=dict(
            title=labels[i],
            titlefont_family="Arial Black",
            tickfont_family="Arial Black",
            titlefont=dict(color=plotly_colors[color_ixs[i]]),
            tickfont=dict(color=plotly_colors[color_ixs[i]])
        ),
        yaxis2=dict(
            title=labels[i + 1],
            titlefont_family="Arial Black",
            tickfont_family="Arial Black",
            titlefont=dict(color=plotly_colors[color_ixs[i + 1]]),
            tickfont=dict(color=plotly_colors[color_ixs[i + 1]]),
            anchor="free",
            overlaying="y",
            side="left",
            position=0.05
        ),
        yaxis3=dict(
            # TODO: remove this
            range=[3.25, 3.4],

            title=labels[i + 2],
            titlefont_family="Arial Black",
            tickfont_family="Arial Black",
            titlefont=dict(color=plotly_colors[color_ixs[i + 2]]),
            tickfont=dict(color=plotly_colors[color_ixs[i + 2]]),
            anchor="x",
            overlaying="y",
            side="right"
        )
    )

    fig.update_yaxes()

    fig.update_layout(barmode='group', title=title)
    fig.show()


def plot_boxes(plot_dict,
               lbls,
               color_ixs=None,
               color_scale=None,
               secondary_y=False,
               plot_title=True,
               title=None,
               label_scale=1,
               boxmode='group',
               x_title=None,
               y_title=None, ):
    color_ixs = list(range(len(plot_dict))) if color_ixs is None else color_ixs

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for j, (k, v) in enumerate(plot_dict.items()):
        x, y = [], []
        for i, y_exp in enumerate(v):
            x += [lbls[i]] * len(y_exp)
            y += y_exp
        fig.add_trace(go.Box(x=x,
                             y=y,
                             name=k.replace('_', ' '),
                             marker_color=plotly_colors[color_ixs[j]],
                             offsetgroup=j + 1,
                             ),
                      secondary_y=j == 1 if secondary_y else False)
    if plot_title:
        fig.update_layout(title=', '.join(plot_dict.keys()) if title is None else title)
    fig.update_layout(boxmode=boxmode, template=template, boxgap=0.1, boxgroupgap=0.1)

    if secondary_y:
        for i in range(len(plot_dict.keys())):
            fig.update_yaxes(dict(color=plotly_colors[color_ixs[i]]),
                             secondary_y=i == 1)  # ,  tickfont_family="Arial Black")
            fig.update_yaxes(title=list(plot_dict.keys())[i].replace('_', ' '),
                             secondary_y=i == 1)  # , titlefont_family="Arial Black")
    else:
        fig.update_yaxes(title=y_title, title_font=dict(size=18 * label_scale, color='black'))
        fig.update_xaxes(tickfont=dict(color='black'))
        fig.update_yaxes(tickfont=dict(color='black'))

    fig.update_xaxes(title=x_title)
    set_fig_font_scale(fig, label_scale)

    fig.show()


def set_fig_font_scale(fig, label_scale):
    fig.update_xaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale, color='black'))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale), title_font=dict(size=18 * label_scale))
    fig.update_layout(legend=dict(font=dict(size=14 * label_scale)))
    fig.update_annotations(font_size=14 * label_scale, font_color='black')


cmaps_names = ['Blues_r', 'Greens_r', 'Oranges_r', 'Reds']


def box_plot_colors(plot_cfg,
                    labels,
                    x_title=None,
                    y_title=None,
                    secondary_y=False,
                    color_label_pos='auto',
                    label_scale=1.8):
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    metrics = plot_cfg['metrics']
    colors = plot_cfg.get('color')
    color_texts = plot_cfg.get('color_text')

    k = len(metrics)
    offsets = (np.arange(k) - np.mean(np.arange(k))) * (1 / k) * 0.95
    cmaps = [matplotlib.cm.get_cmap(cmaps_names[j]) for j in range(len(metrics))]

    global_range = np.max([np.max([np.max(mi) for mi in m]) for m in list(metrics.values())]) - np.min(
        [np.min([np.max(mi) for mi in m]) for m in list(metrics.values())])

    for j, key in enumerate(metrics.keys()):
        metric = metrics[key]

        if colors is not None:
            n = len(metric)
            color = colors[key]
            vmin, vmax = -n, n
        else:
            vmin, vmax = 0, 1
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for i, (lbl, y) in enumerate(zip(labels, metric)):
            fillcolor = 'rgb' + str(cmaps[j](norm(color[i] * 0.7 if colors is not None else 0.5))[0:3])

            fig.add_trace(go.Box(x=np.array([i] * len(y)),
                                 y=y,
                                 name=key.replace('_', ' '),
                                 showlegend=i == 0,
                                 # boxpoints='all',
                                 boxpoints='outliers',  # only outliers
                                 marker_size=3,
                                 fillcolor=fillcolor,
                                 offsetgroup=j + 1,
                                 line=dict(width=3, color='rgb' + str(cmaps[j](vmax)[0:3]))),
                          secondary_y=j == 1 if secondary_y else False)

        if color_texts is not None:
            color_text = color_texts[key]
            y = []

            for m in metric:
                if color_label_pos == 'auto':
                    m_range = np.max(m) - np.min(m) if secondary_y else global_range
                    if (np.quantile(m, 0.75) - np.quantile(m, 0.25)) / m_range > 0.15:
                        y.append(np.median(m))
                    else:
                        y.append(np.max(m) + 0.025 * ((np.max(m) - np.min(m)) if secondary_y else global_range))
                elif color_label_pos == 'max':
                    y.append(np.max(m) + 0.025 * ((np.max(m) - np.min(m)) if secondary_y else global_range))
                elif color_label_pos == 'median':
                    y.append(np.median(m))


            pos_ix = np.where(color_text >= 0)[0]
            neg_ix = np.where(color_text < 0)[0]


            x = np.arange(len(labels)) + offsets[j]

            for ix, color, symbol in zip([pos_ix, neg_ix], ['black', 'red'], ['up', 'down']):

                xi, yi = [x[i] for i in ix], [y[i] for i in ix]
                fig.add_trace(go.Scatter(x=xi,
                                         y=yi,
                                         mode="text+markers",
                                         marker=dict(symbol=f'triangle-{symbol}', size=18, color=color),
                                         text=['<b>{:.2f}</b>'.format(abs(t)) for t in color_text],
                                         textposition='top center',
                                         showlegend=False,
                                         textfont=dict(size=8 * label_scale, color='black'),
                                         ),
                              secondary_y=j == 1 if secondary_y else False)

    fig.update_layout(boxmode='group', template=template, boxgap=0.05, boxgroupgap=0.1)
    fig.update_xaxes(ticktext=labels, tickvals=np.arange(len(labels)))

    if secondary_y:
        for i, key in enumerate(metrics.keys()):
            fig.update_yaxes(dict(color='rgb' + str(cmaps[i](0.5)[0:3])), secondary_y=i == 1)
            fig.update_yaxes(title=key.replace('_', ' '), secondary_y=i == 1)
    else:
        fig.update_yaxes(title=y_title, title_font=dict(size=18 * label_scale, color='black'))
        fig.update_xaxes(tickfont=dict(color='black'))
        fig.update_yaxes(tickfont=dict(color='black'))

    fig.update_xaxes(title=x_title)

    footnote = '<br>'.join(
        [kruskal_significance(metric, label=key)['msg'] for key, metric in plot_cfg['metrics'].items()])
    add_footnote(footnote, fig)
    set_fig_font_scale(fig, label_scale)

    fig.show()


def add_footnote(msg, fig, x=0, y=-0.2):
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=250))

    fig.add_annotation(x=x, y=y,
                       align='left',
                       text=msg,
                       showarrow=False,
                       xanchor='left',
                       xref="paper",
                       yref="paper")


def plot_2d_grouped_traces(points_traces,
                           names=None,
                           markersizes=12,
                           colors=None,
                           modes=None,
                           save=False,
                           save_png=False,
                           file_path=None,
                           title=None,
                           size=(1980, 1080),
                           axes_labels=None,
                           markersymbols=None,
                           centroidsymbols=None,
                           label_scale=1,
                           legend_title=None,
                           ):
    if colors is None:
        colors = DEFAULT_PLOTLY_COLORS[: len(points_traces[0])]

    fig = make_subplots(rows=1, cols=1)
    for i in range(len(points_traces[0])):
        fig.add_trace(go.Scatter(x=points_traces[0][i, :],
                                 y=points_traces[1][i, :],
                                 opacity=0.5,
                                 mode='markers' if modes is None else modes[i],
                                 marker_symbol=None if markersymbols is None else markersymbols[i],
                                 marker=dict(size=markersizes if isinstance(markersizes, int) else markersizes[i],
                                             color=None if colors is None else colors[i]),
                                 showlegend=False,
                                 name=None if names is None else names[i]))

        fig.add_trace(go.Scatter(x=[np.mean(points_traces[0][i, :])],
                                 y=[np.mean(points_traces[1][i, :])],
                                 mode='markers' if modes is None else modes[i],
                                 marker_symbol='x' if centroidsymbols is None else centroidsymbols[i],
                                 marker=dict(
                                     size=markersizes * 2 if isinstance(markersizes, int) else markersizes[i] * 3,
                                     color=None if colors is None else colors[i]),
                                 showlegend=True,
                                 name=None if names is None else names[i]))

        fig.add_shape(type='path',
                      path=confidence_ellipse(x=points_traces[0][i, :],
                                              y=points_traces[1][i, :], n_std=1.5),
                      line={'dash': 'dot'},
                      line_color=None if colors is None else colors[i])

    if axes_labels is not None:
        fig.update_xaxes(title_text=axes_labels[0])
        fig.update_yaxes(title_text=axes_labels[1])

    fig.update_layout(title=title,
                      legend_title_text=legend_title,
                      template=template,
                      font=dict(size=16 * label_scale),
                      legend=dict(font=dict(size=18 * label_scale), orientation="v"),
                      font_color="black")

    fig.update_annotations(font_size=14 * label_scale)
    fig.update_xaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))
    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def plot_radar(data,
               categories,
               names=None,
               markersizes=15,
               colors=None,
               save=False,
               save_png=False,
               file_path=None,
               title=None,
               size=(1980, 1080),
               markersymbols=None,
               label_scale=1,
               legend_title=None,
               ):
    min_radar = np.floor(np.min(data) / 10) * 10

    fig = go.Figure()

    for i in range(data.shape[1]):
        fig.add_trace(go.Scatterpolar(
            r=list(data[:, i]) + [data[0, i]],
            theta=categories + [categories[0]],
            # fill='toself',
            marker_symbol=markersymbols[i],
            marker=dict(size=markersizes, color=None if colors is None else colors[i]),
            line=dict(width=4, color=None if colors is None else colors[i]),
            name=None if names is None else names[i]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min_radar, 100]
            ))
    )

    fig.update_layout(title=title,
                      legend_title_text=legend_title,
                      template=template,
                      font=dict(size=16 * label_scale),
                      legend=dict(font=dict(size=18 * label_scale), orientation="v"),
                      font_color="black")

    fig.update_annotations(font_size=14 * label_scale)
    fig.update_xaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))
    fig.update_yaxes(tickfont=dict(size=14 * label_scale, color='black'), title_font=dict(size=18 * label_scale))
    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)
