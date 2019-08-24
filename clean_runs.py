import datetime
import os

"""
remove folders under runs/ by folder size and date
"""


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


now = datetime.datetime.now().strftime('%b%d')
for dirpath, dirnames, filenames in os.walk('runs2/'):
    for dirname in dirnames:
        path = os.path.join(dirpath, dirname)
        size = get_size(path)
        # print(path, size)
        # rm folder that < 100KB and not today
        if (size < 100000 or 'del' in path) and now not in path:
            print('rm {} with size {}KB'.format(path, size / 1000))
            os.system("rm -rf '{}'".format(path))

    for filename in filenames:
        path = os.path.join(dirpath, filename)
        if 'del' in path and now not in path:
            print(f'rm {path}')
            os.system("rm -rf '{}'".format(path))
    break
