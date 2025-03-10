function bellhopcxx( filename )

% run the BELLHOP program
%
% usage: bellhop( filename )
% where filename is the environmental file

runbellhop = which( 'bellhopcxx.exe' );

if ( isempty( runbellhop ) )
   error( 'bellhopcxx.exe not found in your Matlab path' )
else
   eval( [ '! "' runbellhop '" ' filename ] );
end