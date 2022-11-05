from linetools.spectra.xspectrum1d import XSpectrum1D

from linetools.guis import xspecgui as ltxsg

# fname = '~/Downloads/J0929+4644_172_157_spec.fits'
fname = '~/Downloads/J0943+0531_106_34_redspec.fits'

sp = XSpectrum1D.from_file(fname)

ltxsg.main(sp)
