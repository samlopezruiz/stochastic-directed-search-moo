import os
from datetime import datetime


def get_new_file_path(file_path, extension, use_date_suffix):
    if not isinstance(file_path, list):
        path = os.path.dirname(file_path)
        filename = file_path.split('\\')[-1]
    else:
        path = os.path.join(*file_path[:-1])
        filename = file_path[-1]

    ex = len(extension)
    if use_date_suffix:
        filename = filename + '_' + datetime.now().strftime("%d_%m_%Y %H-%M") + extension
    else:
        filename = filename + extension
        if os.path.exists(os.path.join(path, filename)):
            counter = 1
            filename = '{}_1{}'.format(filename[:-ex], extension)
            while True:
                filename = '{}{}{}'.format(filename[:-(ex + 1)],
                                           str(counter),
                                           extension)
                if not os.path.exists(os.path.join(path, filename)):
                    return os.path.join(path, filename)
                else:
                    counter += 1
        else:
            return os.path.join(path, filename)
    return os.path.normpath(os.path.join(path, filename))


def create_dir(file_path, filename_included=True):
    if not isinstance(file_path, list):
        path = os.path.dirname(file_path) if filename_included else file_path
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        for i in range(1, len(file_path)):
            path = os.path.join(*file_path[:i])
            if not os.path.exists(path):
                os.makedirs(path)