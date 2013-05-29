import os, stat
import subprocess

import numpy

def encode_obshistid(SED_type, filter_name, zenith, seed, redshift):
    SED_types = {'G5v':'1', 'star':'2', 'gal':'3'}
    SED_digit = SED_types[SED_type]
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    redshift_digits = '{:02d}'.format(int(round((redshift / 0.03))))
    return SED_digit + filter_digit + zenith_digit + seed_digit + redshift_digits

def generate_batch_sextract(SED_type, filter_name, zenith, seed, redshift):
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_num = filter_number[filter_name]
    obshistid = encode_obshistid(SED_type, filter_name, zenith, seed, redshift)
    outfilename = 'batch_sextract/batch_sextract_{}.sh'.format(obshistid)

    eimage_filehead = 'eimage_{}_f{}_R22_S11_E000'.format(obshistid, filter_num)
    cat_filehead = '{}_cat'.format(obshistid)

    outputstring = '''OUTPUTDIR=/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/pb12phosim/output/
JOBFILEDIR=`mktemp -d /scratch/$LSB_JOBID.XXXXXX`
echo "Job file directory: $JOBFILEDIR"
echo "Copying job files to $JOBFILEDIR"
cp -p output/{0}.fits.gz $JOBFILEDIR
cp -p sextractor/default.conv $JOBFILEDIR
cp -p sextractor/default.param $JOBFILEDIR
cp -p sextractor/default.psf $JOBFILEDIR
cp -p sextractor/default.sex $JOBFILEDIR
cp -p sextractor/add_noise.py $JOBFILEDIR
cp -p sextractor/measure_second_moments.py $JOBFILEDIR
pushd $JOBFILEDIR
echo "Changing working directory to $JOBFILEDIR"
echo "Adding noise to phosim image"
python add_noise.py {0}.fits.gz
echo "Running SExtractor on noisy image"
sex {0}_noisy.fits -CATALOG_NAME={1}.fits
echo "Computing second moments"
python measure_second_moments.py {0}.fits.gz {1}.fits
echo "gzipping results files"
gzip -f {1}.fits
gzip -f {1}_V.fits
echo "copying results to $OUTPUTDIR"
cp -p {1}.fits.gz $OUTPUTDIR
cp -p {1}_V.fits.gz $OUTPUTDIR
echo "exiting directory $JOBFILEDIR"
popd
echo "Removing $JOBFILEDIR"
rm -Rf $JOBFILEDIR
echo "Success!"
'''.format(eimage_filehead, cat_filehead)
    if not os.path.exists('batch_extract/'):
        os.mkdir('batch_extract')
    with open(outfilename, 'w') as f:
        f.write(outputstring)
        f.close()
    st = os.stat(outfilename)
    os.chmod(outfilename, st.st_mode | stat.S_IEXEC | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def submit_batch_sextract_job(SED_type, filter_name, zenith, seed, redshift):
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_num = filter_number[filter_name]
    obshistid = encode_obshistid(SED_type, filter_name, zenith, seed, redshift)
    if not os.path.exists('stdout/'):
        os.mkdir('stdout/')
    command = 'bsub -q express -oo stdout/{0}sex batch_sextract/batch_sextract_{0}.sh'
    command = command.format(obshistid)
    print command
    subprocess.call(command, shell=True)
