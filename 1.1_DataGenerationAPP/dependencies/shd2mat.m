function shd2mat(filename)
[ PlotTitle, PlotType, freqVec, freq0, atten, Pos, pressure ] =read_shd( filename, 0 );
shdfil = [ filename '.mat' ];   % output file name (pressure)
save( shdfil, 'PlotTitle', 'PlotType', 'freqVec', 'freq0', 'atten', 'Pos', 'pressure' )
end