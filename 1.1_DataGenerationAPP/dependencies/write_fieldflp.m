function write_fieldflp( flpfil, Option, Pos )

% Write a field-parameters file

if ( ~strcmp( flpfil( end - 3 : end ), '.flp' ) )
  flpfil = [ flpfil '.flp' ]; % append extension
end

fid = fopen( flpfil, 'wt' );

fprintf( fid, '/ ! Title\n' );
fprintf( fid, '''%s''  ! Option\n', Option );
fprintf( fid, '999999   ! Mlimit (number of modes to include)\n' );
fprintf( fid, '1        ! NProf \n' );
fprintf( fid, '0.0 /    ! rProf (km)\n' );

% receiver ranges
fprintf( fid, '%5i \t \t \t \t ! NRR\n', length( Pos.r.range ) );

if ( length( Pos.r.range ) > 2 && equally_spaced( Pos.r.range ) )
    fprintf( fid, '    %6f  ', Pos.r.range( 1 ), Pos.r.range( end ) );
else
    fprintf( fid, '    %6f  ', Pos.r.range );
end
fprintf( fid, '/ \t ! RR(1)  ... (km)\n' );

% source depths

fprintf( fid, '%5i \t \t \t \t ! NSD\n', length( Pos.s.z ) );

if ( length( Pos.s.z ) > 2 && equally_spaced( Pos.s.z ) )
    fprintf( fid, '    %6f  ', Pos.s.z( 1 ), Pos.s.z( end ) );
else
    fprintf( fid, '    %6f  ', Pos.s.z );
end

fprintf( fid, '/ \t ! SD(1)  ... (m)\n' );

% receiver depths

fprintf( fid, '%5i \t \t \t \t ! NRD\n', length( Pos.r.z ) );

if ( length( Pos.r.z ) > 2 && equally_spaced( Pos.r.z ) )
    fprintf( fid, '    %6f  ', Pos.r.z( 1 ), Pos.r.z( end ) );
else
    fprintf( fid, '    %6f  ', Pos.r.z );
end

fprintf( fid, '/ \t ! RD(1)  ... (m)\n' );

% receiver range offsets

fprintf( fid, '%5i \t \t ! NRR\n', length( Pos.r.z ) );
fprintf( fid, '    %6.2f  ', zeros( 1, 2 ) );
fprintf( fid, '/ \t \t ! RR(1)  ... (m)\n' );

fclose( fid );
