#include <assert.h>
#include <stdlib.h>

#ifndef PARSE_LIST_H
#define PARSE_LIST_H

struct NumNode
{
	struct NumNode* next_;
	double val_;
};

struct NumList
{
	struct NumNode* head_;
	struct NumNode* tail_;
};

struct NumList* new_numlist (void);

// frees all node in list and reset list head and tail to null
void numlist_clear (struct NumList* list);

void numlist_free (struct NumList* list);

void numlist_pushback (struct NumList* list, double val);


struct PtrNode
{
	struct PtrNode* next_;
	void* val_;
};

struct PtrList
{
	struct PtrNode* head_;
	struct PtrNode* tail_;
	size_t type_;
};

struct PtrList* new_ptrlist (size_t type);

// frees all node in list and reset list head and tail to null
// every val pointer in list passes as a parameter of callback val_mgr
// for memory management
// null val_mgr are ignored (todo: maybe warn?)
void ptrlist_clear (struct PtrList* list, void (*val_mgr)(void*));

// val_mgr is used to clear list before freeing
void ptrlist_free (struct PtrList* list, void (*val_mgr)(void*));

void ptrlist_pushback (struct PtrList* list, void* val);

#endif // PARSE_LIST_H
