#ifndef _LLIST_H
#define _LLIST_H

struct llelem_t {
	struct llelem_t *next;
	char *key;
	void *value;
	size_t size;
} llelem_t;

struct linked_list_t {
	pthread_mutex_t lock;
	struct llelem_t *list;
} linked_list_t;

#endif
