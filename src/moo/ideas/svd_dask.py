if __name__ == '__main__':
    import dask.array as da

    X = da.random.random((200000, 100), chunks=(10000, 100)).persist()

    import dask

    u, s, v = da.linalg.svd(X)
    dask.visualize(u, s, v)
