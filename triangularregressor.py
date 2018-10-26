import numpy as np
import scipy.spatial
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class Triangle(object):
    def __init__(self, vertices):
        """
        Define a 2D triangle spanned by the given vertices.

        It is assumed that the points are not on a line, i.e. that
        (p2 - p1) and (p3 - p1) spanns a plane.

        Arguments
        ---------
        vertices: (3, 2) array of floats
            The 2D coordinates of the three vertices of the triangle.
        """
        try:
            self.vertices = np.asarray(vertices).reshape(3, 2)
        except ValueError:
            raise ValueError("Vertices must be compatible with shape (3, 2)")
        self.p1, self.p2, self.p3 = [p.reshape(-1, 1) for p in self.vertices]
        self.P = np.c_[self.p2 - self.p1, self.p3 - self.p1]

    def barycentric(self, p):
        """
        Return the barycentric coordinates (b1, b2, b3) of p relative
        to the triangle.

        For any point p in the plane spanned by p1, p2, p3 we can write

            b2 (p2 - p1) + b3 (p3 - p1) = p - p1

        Solving this for b2 and b3, setting b1 = 1 - b2 - b3 ensures

            b1 p1 + b2 p2 + b3 p3 = p
              b1  +  b2   +  b3   = 1

        Arguments
        ---------
        p: np.array
            Length-2 array with coordinates for the point.

        Returns
        -------
            Barycentric coordinates of p wrt. the triangle.
        """
        p = p.reshape(-1, 1)
        b = np.linalg.solve(self.P, p - self.p1).ravel()
        return [1 - np.sum(b), b[0], b[1]]

    def __contains__(self, point):
        """Return true if the point is in the triangle."""
        return all(b >= 0 for b in self.barycentric(point))


class Triangulation(object):
    """
    Represent a triangulation in the plane.

    This only acts as a thin convenience wrapper for scipy.spatial.Delaunay.
    """

    def __init__(self, points):
        """
        Create a triangulation from the given array of points.

        Arguments
        ---------
        points: np.array
            An (m x 2) array of m two dimensional points.
        """
        self.simplices = scipy.spatial.Delaunay(points).simplices
        self.triangles = [Triangle(x) for x in points[self.simplices]]

    def containing_triangle(self, point):
        """
        Return the triangle containing point, and the indices to
        said triangle in the original points array.

        Arguments
        ---------
        point: np.array
            Length-2 array with the xy coordinates of the point to find.

        Returns
        -------
            (triangle, indices), where triangle is a (3, 2) array with coordinates
            for the vertices of the containing triangle, and indices is
            the index of the points in the original points array.

            If there is no containing triangle, the return is (None, None)
        """
        for k, triangle in enumerate(self.triangles):
            if point in triangle:
                return triangle, self.simplices[k]
        return None, None


class TriangularRegressor(BaseEstimator, RegressorMixin):
    """
    A piecewise linear regression model for 2D data.

    A triangulation of the provided points is made, and on each triangle a
    linear plane is fitted. The vertices define the domain of the regressors,
    any predictions on points outside the triangulation will be zero.
    """

    def __init__(self, vertices=((0, 0), (1, 0), (0, 1), (1, 1), (.5, .5))):
        """
        Arguments
        ---------
        vertices: array of 2D coordinates, optional
            All vertices to use in the triangulation. Should be a sequence of
            length-2 arrays with xy-coordinates. Default is to use the vertices
            and center of the unit square.
        """
        try:
            self.vertices = np.asarray(vertices).reshape(-1, 2)
        except:
            raise ValueError("Vertices must be interpretable as a (m x 2) array.")

        self.triangulation_ = Triangulation(self.vertices)

    def fit(self, X, y):
        """
        Fit the triangulation of the points in X to the output y.

        Arguments
        ---------
        X: np.array
            (m x 2) array of xy-coordinates.
        y: np.array
            (m x 1) of z-coordinates to fit.

        Returns
        -------
            A reference to self for sklearn compatibility.
        """
        # Verify arguments.
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        if X.shape[1] != 2:
            raise ValueError("Triangulation only implemented for 2D input")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have incompatible dimensions")

        B = self._compute_B(X)
        self.model_ = LinearRegression()
        self.model_.fit(B, y)
        return self

    def _compute_B(self, X):
        B = np.zeros((X.shape[0], self.vertices.shape[0]))
        for k, x in enumerate(X):
            triangle, indices = self.triangulation_.containing_triangle(x)
            if triangle:
                B[k, indices] = triangle.barycentric(x)
        return B

    def _plot_triangles(self):
        plt.triplot(
            self.vertices[:, 0], self.vertices[:, 1], self.triangulation_.simplices
        )
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], "o")

    def predict(self, X):
        """
        Predict z-coordinate for all points in X.

        Arguments
        ---------
        X: array
            (m x 2) array of xy-coordinates.
        Returns
        -------
            (m x 1) of predicted z-coordinates.
        """
        try:
            getattr(self, "model_")
            X = np.asarray(X).reshape(-1, 2)
        except AttributeError:
            raise RuntimeError("You must fit classifier before predicting data!")
        except ValueError:
            raise ValueError(
                "X array must be of shape (m x 2), or be reshape-able to this"
            )

        return self.model_.predict(self._compute_B(X))

    def score(self, X, y=None):
        """
        Returns the coefficient of determination R^2 of the prediction.

        A score function is necessary to be compatible with `GridSearch` and similar.
        """
        return r2_score(y, self.predict(X))


def _franke(x, y):
    """Test function to test interpolation capabilities."""
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def _make_test_data(x, y=None):
    if y is None:
        y = x
    xx, yy = np.meshgrid(x, y)
    zz = _franke(xx, yy)
    X = np.c_[xx.reshape(-1, 1), yy.reshape(-1, 1)]
    return X, zz.reshape(-1, 1), xx, yy, zz


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    np.random.seed(2018)

    vertices = _make_test_data(np.arange(0, 1, 0.1))[0]

    # Make data.
    X_train, z_train, *_ = _make_test_data(*np.random.rand(2, 50))
    X_test, z_test, xx_test, yy_test, zz_test = _make_test_data(*np.random.rand(2, 10))

    # Fit model.
    model = TriangularRegressor(vertices=vertices)
    model.fit(X_train, z_train)
    model._plot_triangles()
    print("Training score:", model.score(X_train, z_train))
    print("Testing score:", model.score(X_test, z_test))

    # 3D Visualization
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    _, _, x_plot, y_plot, z_plot = _make_test_data(np.arange(0, 1, 0.05))
    ax.plot_surface(
        x_plot,
        y_plot,
        z_plot,
        alpha=0.5,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    ax.scatter(xx_test, yy_test, model.predict(X_test).reshape(*xx_test.shape))
    plt.show()
