import warnings

from lsst.sims.catalogs.measures.instance import InstanceCatalog, compound
from lsst.sims.catalogs.generation.db import DBObject, ObservationMetaData
import lsst.sims.catUtils.baseCatalogModels
from lsst.sims.photUtils.EBV import EBVmixin

import numpy

class GalaxyCatalog(InstanceCatalog, EBVmixin):
    comment_char = ''
    catalog_type = 'galaxy_catalog'
    column_outputs = ['galtileid', 'objectId', 'raJ2000', 'decJ2000', 'redshift',
                      'u_ab', 'g_ab', 'r_ab', 'i_ab', 'z_ab', 'y_ab', 'sedPathBulge',
                      'sedPathDisk', 'sedPathAgn', 'magNormBulge', 'magNormDisk', 'magNormAgn',
                      'internalAvBulge', 'internalRvBulge', 'internalAvDisk', 'internalRvDisk',
                      'EBV']
    default_formats = {'S':'%s', 'f':'%.8f', 'i':'%i'}
    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees}

    def get_objectId(self):
        return self.column_by_name(self.refIdCol)

    @compound('sedPathBulge', 'sedPathDisk', 'sedPathAgn')
    def get_sedFilepath(self):
        return (numpy.array([None if k == 'None'
                             else self.specFileMap[k]
                             for k in self.column_by_name('sedFilenameBulge')],
                            dtype=(str, 64)),
                numpy.array([None if k == 'None'
                             else self.specFileMap[k]
                             for k in self.column_by_name('sedFilenameDisk')],
                            dtype=(str, 64)),
                numpy.array([None if k == 'None'
                             else self.specFileMap[k]
                             for k in self.column_by_name('sedFilenameAgn')],
                             dtype=(str, 64)))

obs_metadata = ObservationMetaData(circ_bounds=dict(ra=0.0, dec=0.0, radius=0.05))
dbobj = DBObject.from_objid('galaxyTiled')
constraint = "i_ab < 25.3"
filetype = 'galaxy_catalog'

print "Getting galaxy catalog"
cat = dbobj.getCatalog(filetype, obs_metadata=obs_metadata, constraint=constraint)
filename = "catalog.dat"
print "Writing galaxy catalog"
cat.write_catalog(filename, chunk_size=100000)
