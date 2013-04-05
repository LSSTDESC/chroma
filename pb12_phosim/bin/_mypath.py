import os, sys
thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../lib/')

if libdir not in sys.path:
    sys.path.insert(0, libdir)
