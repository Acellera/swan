#include <pthread.h>

struct __thr_starter_t {
  void *(*func)(void*);
  void *arg;
  int device;
	int thread_index;
	int thread_count;
};

struct __mem_shared_t {
	char *tag;
	void *ptr;
	size_t len;
	struct __mem_shared_t *next;
};



static struct __mem_shared_t *mem_shared = NULL;
static pthread_mutex_t mem_shared_mutex;


static __thread struct __thr_starter_t *thread_data = NULL; 
static pthread_barrier_t global_barrier;

void *swanMallocHostShared( const char *tag, size_t len ) {
	pthread_mutex_lock( &mem_shared_mutex );

	struct __mem_shared_t *p = mem_shared;
	void *memptr = NULL;

	while( p!=NULL ) {
		printf("Searching [%s]for [%s]\n", p->tag, tag );
		if( !strcmp( tag, p->tag )  && (p->len == len ) ) {
			memptr = p->ptr; 
			break; 
		}
		p = p->next;
	}
	if( memptr == NULL ) {
		p = (struct __mem_shared_t*)  malloc( sizeof(struct __mem_shared_t) );
		p->tag = (char*) malloc( strlen( tag ) + 1 );
		memcpy( p->tag, tag, strlen(tag) );
		p->tag[ strlen(tag) ] = '\0';
		p->len = len;
		p->ptr = swanMallocHost( len );
		p->next = mem_shared;
		mem_shared = p;
		memptr = p->ptr;
		printf("New allocation [%s]\n", tag );
	}

	pthread_mutex_unlock( &mem_shared_mutex );

	return memptr;

}

int swanThreadBarrier( void ) {
	swanSynchronize();
	pthread_barrier_wait( &global_barrier );
}

int swanThreadIndex( void ) {
	if( !thread_data  ) { return 0; }
	else return thread_data->thread_index; 
}

int swanThreadCount( void ) {
	if( !thread_data  ) { return 0; }
	else return thread_data->thread_count; 
}

int swanIsParallel( void ) {
	return ( (thread_data!=NULL) && (thread_data->thread_count > 1) ) ;
}


static void* __thr_starter( void *ptrv ) {
  struct __thr_starter_t *ptr = (struct __thr_starter_t*) ptrv;
  swanSetDeviceNumber( ptr->device );
  swanInit();

	thread_data = ptr;

  ptr->func( ptr->arg );
  free( ptr );
  return NULL;
}


swanThread_t * swanInitThread( void*(*func)(void*), void *arg, int* devices, int ndev ) {
	int i=0;
	pthread_t *ptr = (pthread_t*) malloc( sizeof(pthread_t) * ndev );

	pthread_barrier_init( &global_barrier, NULL, ndev );
	pthread_mutex_init( &mem_shared_mutex, NULL );

	for( i=0; i< ndev; i++ ) {
	
		struct __thr_starter_t* aptr = (struct __thr_starter_t*) malloc( sizeof(struct __thr_starter_t) );


		aptr->func     = func;
		aptr->arg      = arg;
		aptr->device   = devices[i];
		aptr->thread_index = i;
		aptr->thread_count = ndev;


		pthread_create( ptr+i, NULL, __thr_starter, (void*)aptr );
	}

	return (swanThread_t *) ptr;
}

void swanWaitForThreads( swanThread_t *thr, int n ) {
	int i;
	for(i=0; i < n; i++ ) {
		pthread_join( (((pthread_t*)thr)[i]), NULL );
	}
}

