#!/usr/bin/env python
#
# shorten.py exerpted and modified from prep_jour.py by M. Fitzgerald. 
# Copyright information for prep_jour.py follows.
# Kyle Barbary, November 2010
#
# Copyright (c) 2007, Michael P. Fitzgerald (mpfitz@berkeley.edu)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Michael P. Fitzgerald may not be used to endorse
#       or promote products derived from this software without specific
#       prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY MICHAEL P. FITZGERALD ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL MICHAEL P. FITZGERALD BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Some of the conversion code ported from J. Baker's nat2jour.pl
# part of the astronat package.
#
# Commands for compressing figures ported from M. Perrin's
# submission-prep Perl script.

import sys, os, re, shutil, string
from optparse import OptionParser

def process_bbl(bbl_infn, bbl_outfn,maxauths):
    "Process the bibliography file.  Returns a list of bibliography keys in order of bib. entry."

    bbl_inf = open(bbl_infn)

    procbib = False # currently processing bibitems
    bibinfo = {} # contains bibliography information for each key
    bibkeys = [] # ordered list of keys
    n_auths = {} # number of authors

    for line in bbl_inf:
        # new bibitem
        if re.match(r"\\bibitem", line):
            item = line
            procbib = True
            continue

        # append line to current item strint
        if procbib:
            item += line

        # end of item -- process
        if re.match(r"\n", line) and procbib:
            procbib = False

            # strip newlines
            item = re.sub(r"\n", '', item)

            # strip '\natexlab's
            item = re.sub(r"\{\\natexlab\{(.*?)\}\}", r"\1", item)

            # strip \noopsort{}'s
            item = re.sub(r"\{\\noopsort\{(.*?)\}\}", '', item)

            # parse entry
            m = re.match(r"\\bibitem\[\{(.*?)\((\d{4}[a-z]*)\)(.*?)\}\]\{(.*?)\}(.*)", item)
            if m:
                shortlist, year, longlist, key, ref = m.groups()
                authlist = longlist and longlist or shortlist

                # save number of authors
                n_auths[key] = authlist.count(',')+1

                # shorten author list in reference if too long
                if n_auths[key] > maxauths:
                    # extract author list, last comma before year (if any) and year
                    m = re.match(r"(.*?)(\,)?\s*(\d{4})", ref)
                    if not m:
                        print "Can't find year in reference:\n%s" % ref
                    auths, lastcomma = m.group(1, 2)
                    lastcomma = lastcomma or '' # set blank string if no match
                    end = m.start(3) # position of year

                    if not re.match(r"\w", auths): # avoid refs like "---, 1998, ..."
                        #n_commas = maxauths*2
                        n_commas = 2
                        if n_commas > auths.count(','):
                            print "Error trying to truncate author list:\n%s" % ref
                        else:
                            # find the index of the Nth comma
                            pos = 0
                            for i in range(n_commas):
                                pos = auths.index(',', pos)
                                pos += 1
                            # replace with etal
                            ref = ref[0:pos]+" {et~al.}"+lastcomma+ref[end-1:]

                # output entry
                bibkeys.append(key)
                bibinfo[key] = shortlist, year, longlist, ref

            else: # didn't match
                print "weird bibitem: %s" % item

    bbl_inf.close()

    # output info to bbl file
    bbl_outf = open(bbl_outfn, 'w')
    bbl_outf.write("\\begin{thebibliography}{%d}\n\n" % len(bibkeys))
    for key in bibkeys:
        shortlist, year, longlist, ref = bibinfo[key]
        bbl_outf.write("\\bibitem[{%s(%s)%s}]{%s}\n%s\n\n" % (shortlist, year, longlist, key, ref))
    bbl_outf.write("\\end{thebibliography}\n")
    bbl_outf.close()

    return bibkeys

#################
#     MAIN      #
#################
if __name__ == '__main__':

    usage="usage: %prog [options] filename"
    description="Shorten author lists in bbl file. Save to filename.short"

    parser = OptionParser(usage=usage)
    parser.add_option("-n", "--number", dest="maxauth", type="int", default=8,
                      help="Maximum number of authors to allow in bbl file.")
    (options, args) = parser.parse_args()

    bbl_infn=args[0]
    bbl_outfn=bbl_infn+'.short'
    process_bbl(bbl_infn,bbl_outfn,options.maxauth)
