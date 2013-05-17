import subprocess

def encode_obshistid(SED_type, filter_name, zenith, seed, redshift):
    SED_types = {'G5v':'1', 'star':'2', 'gal':'3'}
    SED_digit = SED_types[SED_type]
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    redshift_digits = '{:02d}'.format(int(round((redshift / 0.03))))
    return SED_digit + filter_digit + zenith_digit + seed_digit + redshift_digits

def submit_job(SED_type, filter_name, zenith, seed, redshift, test=False):
    obshistid = encode_obshistid(SED_type, filter_name, zenith, seed, redshift)
    phosim_dir = '/nfs/slac/g/ki/ki19/jmeyers3/phosim-3.2.9/'
    sub_dir = '/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/pb12phosim/'
    cat_dir = sub_dir + 'catalogs/'
    cat_file = cat_dir + 'stargrid_' + obshistid
    extra_file = cat_file + '_extra'
    out_dir = '/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/pb12phosim/output/'
    command = '"cd {} && ./phosim {} -c {} -e0 -sR22_S11 -o {}"'.format(phosim_dir,
                                                                       cat_file,
                                                                       extra_file,
                                                                       out_dir)
    stdout_file = sub_dir + 'stdout/' + obshistid
    job_name = obshistid
    full_command = 'bsub -q xlong -oo {} -J {} {}'.format(stdout_file, job_name, command)
    print full_command
    if not test:
        subprocess.call(full_command, shell=True)

if __name__ == '__main__':
    import numpy as np
    for z in np.arange(0.0, 3.0, 0.03):
        submit_job('gal', 'r', 30.0, 1000, z)
        submit_job('gal', 'i', 30.0, 1000, z)
    submit_job('star', 'r', 30.0, 1000, 0.0)
    submit_job('star', 'i', 30.0, 1000, 0.0)
    submit_job('G5v', 'r', 30.0, 1000, 0.0)
    submit_job('G5v', 'i', 30.0, 1000, 0.0)
