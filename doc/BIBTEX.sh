bibtex $1
python shorten.py -n3 $1.bbl
mv -f $1.bbl.short $1.bbl
