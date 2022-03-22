import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np
from plotly.subplots import make_subplots
from src.utils.files import create_dir, get_new_file_path

pio.renderers.default = "browser"

colors = [
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
                                overlayed=False,  # when this fig will be overlayed on top
                                pareto_marker_mode='lines'):
    c_points, c_uv, c_xy, p_points, p_uv, p_xy = get_arrows(correctors, points, predictors)

    fig = make_subplots(rows=1, cols=1)

    if pareto is not None and not overlayed:
        pareto_plot = pareto.to_numpy() if isinstance(pareto, pd.DataFrame) else pareto
        fig.add_trace(go.Scatter(x=pareto_plot[:, 0],
                                 y=pareto_plot[:, 1],
                                 marker=dict(color=colors[0]),
                                 mode=pareto_marker_mode,
                                 showlegend=not overlayed,
                                 name='pareto front'))

    if plot_arrows:
        p_quiver_fig = ff.create_quiver(p_xy[:, 0], p_xy[:, 1], p_uv[:, 0], p_uv[:, 1],
                                        scale=scale,
                                        arrow_scale=arrow_scale,
                                        showlegend=not overlayed,
                                        name='predictor',
                                        marker=dict(color=colors[3]),
                                        line_width=line_width)

        fig.add_traces(data=p_quiver_fig.data)

        c_quiver_fig = ff.create_quiver(c_xy[:, 0], c_xy[:, 1], c_uv[:, 0], c_uv[:, 1],
                                        scale=scale,
                                        arrow_scale=arrow_scale,
                                        showlegend=not overlayed,
                                        name='corrector',
                                        marker=dict(color=colors[1]),
                                        line_width=line_width)

        fig.add_traces(data=c_quiver_fig.data)

    if plot_points:
        fig.add_trace(go.Scatter(x=p_points[:, 0],
                                 y=p_points[:, 1],
                                 mode='markers',
                                 showlegend=not overlayed and not plot_arrows,
                                 marker=dict(size=markersize,
                                             color=colors[3]),
                                 name='predictor'))

        if c_points.shape[0] > 0:
            fig.add_trace(go.Scatter(x=c_points[:, 0],
                                     y=c_points[:, 1],
                                     mode='markers',
                                     showlegend=not overlayed and not plot_arrows,
                                     marker=dict(size=markersize,
                                                 color=colors[1]),
                                     name='corrector'))

    fig.add_trace(go.Scatter(x=points[:, 0],
                             y=points[:, 1],
                             mode='markers',
                             showlegend=not overlayed,
                             marker=dict(size=markersize * 1.5,
                                         color='rgba(1, 1, 1, 1)'),
                             name=point_name))

    if not overlayed:
        fig.add_trace(go.Scatter(x=points[:1, 0],
                                 y=points[:1, 1],
                                 mode='markers',
                                 marker_symbol='x',
                                 marker=dict(size=markersize * 3,
                                             color='rgba(1, 1, 1, 1)'),
                                 name='initial'))

    if return_fig:
        return fig

    fig.update_layout(title=title)
    fig.show()

    if file_path is not None and save is True:
        plotly_save(fig, file_path, size, save_png=save_png)


def get_arrows(correctors, points, predictors):
    p_xy, p_uv, p_points, c_uv, c_xy, c_points = [], [], [], [], [], []
    # predictors
    for corrs, preds, point in zip(correctors, predictors, points[:-1]):
        for p in preds:
            p_xy.append(point)
            p_uv.append(p - point)
            p_points.append(p)

        c_last = preds[0]
        for i, c in enumerate(corrs):
            if i > 0:
                c_points.append(c)
            c_xy.append(c_last)
            c_uv.append(c - c_last)
            c_last = c
    p_xy, p_uv, p_points = np.array(p_xy), np.array(p_uv), np.array(p_points)
    c_points, c_uv, c_xy = np.array(c_points), np.array(c_uv), np.array(c_xy)
    return c_points, c_uv, c_xy, p_points, p_uv, p_xy


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
                          ):
    fig = make_subplots(rows=1, cols=1)

    for i, points in enumerate(points_traces):
        fig.add_trace(go.Scatter(x=points[:, 0],
                                 y=points[:, 1],
                                 mode='markers' if modes is None else modes[i],
                                 marker=dict(size=markersizes if isinstance(markersizes, int) else markersizes[i],
                                             color=None if color_ixs is None else colors[color_ixs[i]]),
                                 name=None if names is None else names[i]))

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
                                 pareto,
                                 arrow_scale=0.4,
                                 markersize=12,
                                 save=False,
                                 save_png=False,
                                 file_path=None,
                                 size=(1980, 1080),
                                 plot_arrows=True,
                                 plot_points=True,
                                 plot_ps=True,
                                 pareto_marker_mode='lines',
                                 return_fig=False,
                                 ):
    results = [results] if not isinstance(results, list) else results
    pred_figs, corr_figs = [], []
    titles = ['Continuation method: predictors and correctors in decision space',
              'Continuation method: predictors and correctors in objective space']
    for i, res in enumerate(results):
        if plot_ps:
            pred_fig = plot_2D_predictor_corrector(points=res['X'],
                                                   predictors=res['X_p'],
                                                   correctors=res['X_c'],
                                                   pareto=pareto['ps'],
                                                   title=titles[i],
                                                   arrow_scale=arrow_scale,
                                                   point_name='x',
                                                   markersize=markersize,
                                                   line_width=1.5,
                                                   plot_arrows=plot_arrows,
                                                   plot_points=plot_points,
                                                   return_fig=True,
                                                   overlayed=i > 0,
                                                   save=False,
                                                   pareto_marker_mode=pareto_marker_mode
                                                   )

        corr_fig = plot_2D_predictor_corrector(points=res['F'],
                                               predictors=res['F_p'],
                                               correctors=res['F_c'],
                                               pareto=pareto['pf'],
                                               arrow_scale=arrow_scale,
                                               point_name='f(x)',
                                               markersize=markersize,
                                               line_width=1.5,
                                               plot_arrows=plot_arrows,
                                               plot_points=plot_points,
                                               return_fig=True,
                                               overlayed=i > 0,
                                               save=False,
                                               pareto_marker_mode=pareto_marker_mode
                                               )
        corr_figs.append(corr_fig)
        if plot_ps:
            pred_figs.append(pred_fig)

    if plot_ps:
        fig = make_subplots(rows=1, cols=1)
        [fig.add_traces(data=f.data) for f in pred_figs]
        fig.show()

        if file_path is not None and save is True:
            plotly_save(fig, file_path + '_ds', size, save_png=save_png)

    fig = make_subplots(rows=1, cols=1)
    [fig.add_traces(data=f.data) for f in corr_figs]
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
                                           line=dict(color=colors[4]),
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
                                         line=dict(color=colors[4]),
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
                                         color=colors[color_ix[0]]),
                             mode='markers',
                             name=points_name))

    if centers is not None:
        fig.add_trace(go.Scatter(x=centers[:, 0],
                                 y=centers[:, 1],
                                 showlegend=True,
                                 marker=dict(size=markersize,
                                             color=colors[color_ix[1]]),
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


def plot_points_3d(points,
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
    fig = make_subplots(rows=1, cols=1)
    if not only_best:
        fig.add_trace(go.Scatter3d(x=points[:, 0],
                                   y=points[:, 1],
                                   z=points[:, 2],
                                   showlegend=True,
                                   marker=dict(size=markersize / 1.5,
                                               color=colors[color_ix[0]]),
                                   opacity=0.7 if secondary is not None and not only_best else 1,
                                   mode='markers',
                                   name=points_name))

    if secondary is not None:
        fig.add_trace(go.Scatter3d(x=secondary[:, 0],
                                   y=secondary[:, 1],
                                   z=secondary[:, 2],
                                   showlegend=True,
                                   marker=dict(size=markersize,
                                               color=colors[color_ix[1]]),
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
