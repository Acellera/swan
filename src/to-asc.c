#include <stdio.h>
#include <stdlib.h>

int main(int argc, char**argv ) {

	int i=0;
	FILE *fin;
	if(argc==2 ) {
	fin = fopen(argv[1], "rb" );
	if(fin==NULL) { exit(0); }
		printf("{\n");
	while(!feof(fin) ) {
		unsigned char b;
		fread( &b, 1, 1, fin );
		printf("0x%02x, ", b );
		i++;
		if( i%16==0 ) { printf("\n"); }
		
	}
	printf("0x00 };\n\n" );
	}

}	
