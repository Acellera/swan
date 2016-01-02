#include "swan_api.h"

int main(void) {
	swanEnumerateDevices( stdout );

	printf("THIS DEVICE HAS %d CEs\n", swanGetNumberOfComputeElements() );
}

