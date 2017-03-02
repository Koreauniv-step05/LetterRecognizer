import os

def load_allpath(path):
    res = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file[0] == '.':
                print("ignore %s" % os.path.join(root, file))
                continue
            else:
                filepath = os.path.join(root, file)
                res.append(filepath)
    return res

def get_folder_name(path):
    return os.path.basename(os.path.dirname(path))
