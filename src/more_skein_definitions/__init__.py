from os.path import dirname, basename, isfile, join
import glob
modules = glob.glob(join(dirname(__file__), "*.py"))
[__import__(f"src.more_skein_definitions.{basename(f)[:-3]}", locals(), globals())
           for f in modules if isfile(f) and not f.endswith('__init__.py')]
# Taken from https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder
