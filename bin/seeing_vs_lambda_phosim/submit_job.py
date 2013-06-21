import os
import subprocess

import numpy

def encode_obshistid(mono_wave, filter_name, zenith, seed):
    #first four digits are wavelength in nm
    wave_digits = str(int(round(mono_wave)))
    filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}
    filter_digit = filter_number[filter_name]
    zenith_digit = str(int(round((zenith / 10.0))))
    seed_digit = str(seed - 1000)
    return wave_digits + filter_digit + zenith_digit + seed_digit

def submit_job(mono_wave, filter_name, zenith, seed, test=False):
    obshistid = encode_obshistid(mono_wave, filter_name, zenith, seed)
    phosim_dir = '/nfs/slac/g/ki/ki19/jmeyers3/phosim-3.2.9/'
    sub_dir = '/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/seeing_vs_lambda_phosim/'
    cat_dir = sub_dir + 'catalogs/'
    cat_file = cat_dir + 'stargrid_' + obshistid
    extra_file = cat_file + '_extra'
    if not os.path.exists('output/'):
        os.mkdir('output/')
    if not os.path.exists('stdout/'):
        os.mkdir('stdout/')
    out_dir = '/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/seeing_vs_lambda_phosim/output/'
    command = '"cd {} && ./phosim {} -c {} -e0 -sR22_S11 -o {}"'.format(phosim_dir,
                                                                       cat_file,
                                                                       extra_file,
                                                                       out_dir)
    stdout_file = sub_dir + 'stdout/' + obshistid
    job_name = obshistid
    full_command = 'bsub -q long -oo {} -J {} {}'.format(stdout_file, job_name, command)
    print full_command
    if not test:
        subprocess.call(full_command, shell=True)

if __name__ == '__main__':
    for w in numpy.arange(300, 401, 25):
        submit_job(w, 'u', 0, 1000)
    for w in numpy.arange(400, 551, 25):
        submit_job(w, 'g', 0, 1000)
    for w in numpy.arange(550, 701, 25):
        submit_job(w, 'r', 0, 1000)
    for w in numpy.arange(675, 826, 25):
        submit_job(w, 'i', 0, 1000)
    for w in numpy.arange(800, 951, 25):
        submit_job(w, 'z', 0, 1000)
    for w in numpy.arange(900, 1101, 25):
        submit_job(w, 'Y', 0, 1000)
