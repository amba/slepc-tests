#!/usr/bin/env gnuplot
set term qt persist

set xlabel 'φ / π'
set ylabel 'E / Δ'
set grid

plot for [i=2:32] 'output-spin.dat' using ($1/pi):i notitle
pause mouse close