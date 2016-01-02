#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include "list.h"




void ll_init( struct linked_list_t *l ) {
	memset( l, 0, sizeof(struct linked_list_t) );
	pthread_mutex_init( &(l->lock), NULL );
}


void ll_put( struct linked_list_t *l, char *key, void *value, size_t len ) {
	pthread_mutex_lock( &(l->lock) );
	struct llelem_t *p = (struct llelem_t*) malloc( sizeof( struct llelem_t) );
	p->key = (char*) malloc( strlen(key) + 1 );
	strncpy( p->key, key, strlen(key) );
	p->value = value;
	p->size   = len;
	p->next  = l->list;
	l->list  = p;
	pthread_mutex_unlock( &(l->lock) );
}

void * ll_get( struct linked_list_t *l, char *key, size_t *size ) {
	void *retval=NULL;
	*size = 0;
	pthread_mutex_lock( &(l->lock) );
	
	struct llelem_t *p = l->list;
	while( p!=NULL ) {
		if(!strcmp( p->key, key ) ) {
			*size = p->size;
			retval = p->value;
		}
		p = p->next;
	}

	pthread_mutex_unlock( &(l->lock) );

	return retval;

}


