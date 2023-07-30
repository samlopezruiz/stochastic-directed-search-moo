from copy import copy, deepcopy

import numpy as np
from numpy.linalg import norm
from plotly.subplots import make_subplots

from src.utils.plot import plot_boxes_2d, plot_points_centers_2d


class Node:
    def __init__(self, limits):
        self.limits = limits
        self.c = np.sum(limits, axis=1) / 2
        self.left = None
        self.right = None
        self.fxs = []
        self.xs = []
        self.dxs = []
        self.verteces = []
        self.min_dist = None
        self.best_ix = None

    def create_left_child(self, axis):
        limits = copy(self.limits)
        limits[axis, 1] = self.c[axis]  # ub = c
        self.left = Node(limits)

    def create_right_child(self, axis):
        limits = copy(self.limits)
        limits[axis, 0] = self.c[axis]  # lb = c
        self.right = Node(limits)

    def append_to(self, fx, x=None, dx=None):
        if len(self.fxs) > 0:
            if not np.any(fx > self.fxs[self.best_ix]):
                self.min_dist = norm(fx - self.c)
                self.best_ix = len(self.fxs)
            elif norm(fx - self.c) < self.min_dist:
                self.min_dist = norm(fx - self.c)
                self.best_ix = len(self.fxs)

        else:
            self.min_dist = norm(fx - self.c)
            self.best_ix = 0

        self.fxs.append(fx)
        self.xs.append(x)
        self.dxs.append(dx)

    def has_points(self):
        return len(self.fxs) > 0

    def get_boxes_edges(self):
        vertices = self.get_vertices()
        edges = []
        for i in range(len(vertices)):
            u = vertices[i]
            for v in vertices[i + 1:]:
                if np.sum(np.abs(v - u) > 0) <= 1:
                    edges.append((u, v))

        return edges

    def get_vertices(self):
        old = [[]]
        for axis in range(self.limits.shape[0]):
            all = []
            for lim in self.limits[axis, :]:
                new = deepcopy(old)
                [new[i].append(lim) for i in range(len(new))]
                all += new
            old = all

        return np.array(all)


class Boxes:

    def __init__(self, max_h, limits):
        self.max_h = max_h
        self.limits = limits
        self.k = limits.shape[0]
        self.root = Node(limits)
        self.boxes = [[]]

    def has_box(self, fx):
        node = self.root

        # fx is outside the limits
        if np.any(fx > self.limits[:, 1]) or np.any(fx < self.limits[:, 0]):
            return None

        for h in range(self.max_h - 1):
            axis = h % self.k
            if fx[axis] < node.c[axis]:
                if node.left:
                    node = node.left
                else:
                    return False

            else:
                if node.right:
                    node = node.right
                else:
                    return False

        return True

    def insert(self, fx, x=None, dx=None):
        node = self.root
        created = False

        if not np.all(fx >= self.limits[:, 0]) or not np.all(fx >= self.limits[:, 0]):
            return None, False

        for h in range(self.max_h - 1):
            axis = h % self.k
            if fx[axis] < node.c[axis]:
                if not node.left:
                    node.create_left_child(axis)
                    created = True
                node = node.left

            else:
                if not node.right:
                    node.create_right_child(axis)
                    created = True
                node = node.right

        node.append_to(fx, x, dx)

        return node, created

    def get_boxes(self):
        self.boxes = [[] for _ in range(self.max_h)]
        self._get_boxes(self.root, 0)
        return self.boxes

    def _get_boxes(self, node, h):
        self.boxes[h].append(node.get_boxes_edges())
        if node.left:
            self._get_boxes(node.left, h + 1)
        if node.right:
            self._get_boxes(node.right, h + 1)

    def get_points(self):
        self.fxs = []
        self.xs = []
        self.cs = []
        self.best_i = []
        self._get_points(self.root)
        return {'fx': np.array(self.fxs), 'x': np.array(self.xs), 'c': np.array(self.cs), 'best_ix': self.best_i}

    def _get_points(self, node):
        if node.has_points():
            [self.fxs.append(fx) for fx in node.fxs]
            [self.xs.append(x) for x in node.xs]
            [self.best_i.append(i == node.best_ix) for i, _ in enumerate(node.fxs)]
            self.cs.append(node.c)

        if node.left:
            self._get_points(node.left)
        if node.right:
            self._get_points(node.right)


if __name__ == '__main__':

    # Example of boxes and points
    limits = np.array([[0, 10]] * 3).astype(np.float)

    boxes = Boxes(8, limits)

    node, created = boxes.insert(np.array([8.2, 8.9, 8.]))
    node, created = boxes.insert(np.array([8., 8., 8.]))
    node, created = boxes.insert(np.array([8.7, 8.1, 8.]))
    node, created = boxes.insert(np.array([4., 1., 1.]))

    boxes_edges = boxes.get_boxes()
    points = boxes.get_points()

    edg_fig = plot_boxes_2d(boxes_edges, return_fig=True)

    pts_fig = plot_points_centers_2d(points['fx'],
                                     centers=points['c'],
                                     best=points['best_ix'],
                                     return_fig=True)
    fig_data = [edg_fig.data, pts_fig.data]

    fig = make_subplots()
    for data in fig_data:
        fig.add_traces(data)

    fig.show()
