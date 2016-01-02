void swanEnableCheckpointing( void ) {
  if( state.init ) {
    error( "Cannot enable checkpointing once Swan is initialised" );
  }
  state.checkpointing = 1;
}

