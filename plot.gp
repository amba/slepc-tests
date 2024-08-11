#!/usr/bin/env gnuplot
set term qt persist

set xlabel 'φ / π'
set ylabel 'E / Δ'
set grid
# set palette defined (0 'blue', 1 'red')
#set cbrange [9e7:]
plot for [i=3:23] 'output-spin.dat' using ($2/pi):i:1 palette pt 7 ps 1 notitle
pause mouse close