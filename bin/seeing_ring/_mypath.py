import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
libdir = os.path.abspath(os.path.join(thisdir, '../../'))

if libdir not in sys.path:
    sys.path.insert(0, libdir)
