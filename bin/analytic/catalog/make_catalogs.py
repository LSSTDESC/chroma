""" Script authored principally by Simon Krughoff to use LSST catalogs framework to create
text file star and galaxy catalogs from LSST CatSim.  These can be used to determine a
realistic distribution of chromatic biases.  This script requires that the LSST catalog
framework is accessible.
"""

import os
from argparse import ArgumentParser
import warnings

from lsst.sims.catalogs.measures.instance import InstanceCatalog, compound
from lsst.sims.catalogs.generation.db import DBObject, ObservationMetaData
import lsst.sims.catUtils.baseCatalogModels
from lsst.sims.photUtils.EBV import EBVmixin
from lsst.sims.coordUtils import AstrometryBase

import numpy

#First define the catalogs to output.  Since they have different columns
#the galaxy and star catalogs need to have different definitions.
#This is (hopefully) pretty straight forward.  Column names can be either
#the names listed in the model file:
#$CATALOGS_GENERATION_DIR/python/lsst/sims/catalogs/generation/db/GalaxyModels.py
#$CATALOGS_GENERATION_DIR/python/lsst/sims/catalogs/generation/db/StarModels.py
#or the column names from the database.  You can get all possible column names
#by doing:
#>>> dbobj = DBObject.from_objid('galaxyTile') #or any other defined object type
#>>> dbobj.show_mapped_columns()

class ExampleGalaxyCatalog(InstanceCatalog, EBVmixin, AstrometryBase):
# class ExampleGalaxyCatalog(InstanceCatalog):
    comment_char = ''
    catalog_type = 'example_galaxy_catalog'
    column_outputs = ['galtileid', 'objectId', 'raJ2000', 'decJ2000', 'redshift',
                      'a_b', 'a_d', 'a_d', 'b_d', 'pa_bulge', 'pa_disk',
                      'u_ab', 'g_ab', 'r_ab', 'i_ab', 'z_ab', 'y_ab',
                      'sedPathBulge', 'sedPathDisk', 'sedPathAgn',
                      'magNormBulge', 'magNormDisk', 'magNormAgn',
                      'internalAvBulge', 'internalRvBulge', 'internalAvDisk', 'internalRvDisk',
                      'glon', 'glat', 'EBV']
    default_formats = {'S':'%s', 'f':'%.8f', 'i':'%i'}
    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees}

    def get_objectId(self):
        return self.column_by_name(self.refIdCol)
    #The compound decorator is a special decorator that allows the catalogs
    #framework to find the columns for output when several columns are generated
    #in the same method.
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

class ExampleStarCatalog(InstanceCatalog):
    comment_char = ''
    catalog_type = 'example_star_catalog'
    column_outputs = ['objectId', 'raJ2000', 'decJ2000', 'magNorm',
                      'umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag', 'sedFilepath',
                      'galacticAv']
    default_formats = {'S':'%s', 'f':'%.8f', 'i':'%i'}
    transformations = {'raJ2000':numpy.degrees, 'decJ2000':numpy.degrees}

    def get_objectId(self):
        return self.column_by_name(self.refIdCol)

    def get_sedFilepath(self):
        return numpy.array([self.specFileMap[k] for k in self.column_by_name('sedFilename')])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ra', type=float, default=200.,
                        help="RA in degrees of center of pointing")
    parser.add_argument('--dec', type=float, default=-10.,
                        help="Dec in degrees of center of pointing")
    parser.add_argument('--size', type=float, default=1.0,
                        help="Size of the pointing in degrees")
    parser.add_argument('--verbose', action="store_true",
                        help="Show all warnings")
    args = parser.parse_args()

    if not args.verbose:
        warnings.simplefilter("ignore")
    #Get the observation data.  This is ra, dec, and radius in degrees
    #You'll notice that the galaxies come back in a circular aperture and the
    #stars come back in a RA/Dec box.  Sorry about that bug.  I'll have it fixed
    #shortly.
    obs_metadata = ObservationMetaData(circ_bounds=dict(ra=args.ra, dec=args.dec, radius=args.size))
    #Get the database object by name:
    #You can get the names of the defined object types by doing:
    #>>> print DBObject
    dbobj = DBObject.from_objid('galaxyTiled')
    #Constraint in SQL
    constraint = "i_ab < 25.3"
    #File type: This is the value of catalog_type in the catalog definitions above
    filetype = 'example_galaxy_catalog'

    #Get the catalog
    print "Getting galaxy catalog"
    cat = dbobj.getCatalog(filetype, obs_metadata=obs_metadata, constraint=constraint)
    #Write the catalog out in chunks.  I find that 100000 is a good number of lines to
    #write at a time.  It is not too slow but will not push on memory limits.
    if not os.path.isdir('output/'):
        os.mkdir('output/')
    filename = "output/galaxy_catalog.dat"
    print "Writing galaxy catalog"
    cat.write_catalog(filename, chunk_size=100000)

    #Same as above but for main sequence stars.
    dbobj = DBObject.from_objid('allstars')
    constraint = "imag < 22 and imag > 16"
    filetype = 'example_star_catalog'
    print "Getting star catalog"
    cat = dbobj.getCatalog(filetype, obs_metadata=obs_metadata, constraint=constraint)
    print "Writing star catalog"
    filename = "output/star_catalog.dat"
    cat.write_catalog(filename, chunk_size=100000)
