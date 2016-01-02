#include "gridblock.kh"


int main(void) {
	block_config_t grid, block;
	int max;
	int sum;
	uint tmp;
	uint *ptr = (uint*) swanMalloc( sizeof(uint) );

	grid.x = 2;
	grid.y = 3;
	grid.z = 1;
	block.x = 4;
	block.y = 5;
	block.z = 2;

	max = grid.x * grid.y * grid.z * block.x * block.y * block.z;

	k_gridblock( grid, block, 0, ptr );

	swanMemcpyDtoH( ptr, &tmp, sizeof(uint) );

	sum = (max*max + max) /2;

	if( sum != tmp )  {
		printf("FAILED: Expected: %d , computed: %d\n", sum, tmp );
	}
	else {
		printf("SUCCESS\n");
	}

}
