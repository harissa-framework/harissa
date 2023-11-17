import os
import sys
from importlib.metadata import version as get_version

root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))

version = get_version('harissa')
filename = \
    os.path.join(root_dir, sys.argv[1], 'index.html') if len(sys.argv)>1 else\
    os.path.join(root_dir, 'index.html')

with open(filename, 'w') as file:
    file.write(
f'''<!DOCTYPE html>
<html>
  <head>
    <title>Redirecting to v{version}</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=./v{version}/index.html">
    <link rel="canonical" href="harissa-framework.github.io/harissa/v{version}/index.html">
  </head>
</html>
''')