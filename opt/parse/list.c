#include <string.h>

#include "opt/parse/list.h"

#ifdef PARSE_LIST_H

struct NumList* new_numlist (void)
{
	size_t nlistbytes = sizeof(struct NumList);
	struct NumList* out = malloc(nlistbytes);
	memset(out, 0, nlistbytes);
	return out;
}

void numlist_move (struct NumList* dst, struct NumList* src)
{
	if (NULL == dst || NULL == src)
	{
		return;
	}
	memcpy(dst, src, sizeof(struct NumList));
	src->head_ = src->tail_ = NULL;
}

void numlist_clear (struct NumList* list)
{
	if (NULL == list)
	{
		return;
	}
	struct NumNode* it = list->head_;
	struct NumNode* next;
	// free every node
	while (NULL != it)
	{
		next = it->next_;
		free(it);
		it = next;
	}
	list->head_ = list->tail_ = NULL;
}

void numlist_free (struct NumList* list)
{
	if (NULL == list)
	{
		return;
	}
	numlist_clear(list);
	free(list);
}

void numlist_pushback (struct NumList* list, double val)
{
	if (NULL == list)
	{
		return;
	}
	assert(NULL != list->head_ || list->head_ == list->tail_);
	size_t nnodebytes = sizeof(struct NumNode);
	if (NULL == list->tail_)
	{
		list->head_ = list->tail_ = malloc(nnodebytes);
	}
	else
	{
		list->tail_ = list->tail_->next_ = malloc(nnodebytes);
	}
	list->tail_->next_ = NULL;
	list->tail_->val_ = val;
}


struct PtrList* new_ptrlist (size_t type)
{
	size_t nlistbytes = sizeof(struct PtrList);
	struct PtrList* out = malloc(nlistbytes);
	memset(out, 0, nlistbytes);
	out->type_ = type;
	return out;
}

void ptrlist_move (struct PtrList* dst, struct PtrList* src)
{
	if (NULL == dst || NULL == src)
	{
		return;
	}
	memcpy(dst, src, sizeof(struct PtrList));
	src->head_ = src->tail_ = NULL;
	dst->type_ = src->type_;
}

void ptrlist_clear (struct PtrList* list, void (*val_mgr)(void*))
{
	if (NULL == list)
	{
		return;
	}
	assert(NULL != list->head_ || list->head_ == list->tail_);
	struct PtrNode* it = list->head_;
	struct PtrNode* next;
	// free every node
	while (NULL != it)
	{
		next = it->next_;
		if (NULL != val_mgr)
		{
			val_mgr(it->val_);
		}
		free(it);
		it = next;
	}
	list->head_ = list->tail_ = NULL;
}

void ptrlist_free (struct PtrList* list, void (*val_mgr)(void*))
{
	if (NULL == list)
	{
		return;
	}
	ptrlist_clear(list, val_mgr);
	free(list);
}

void ptrlist_pushback (struct PtrList* list, void* val)
{
	if (NULL == list)
	{
		return;
	}
	assert(NULL != list->head_ || list->head_ == list->tail_);
	size_t nnodebytes = sizeof(struct PtrNode);
	if (NULL == list->tail_)
	{
		list->head_ = list->tail_ = malloc(nnodebytes);
	}
	else
	{
		list->tail_ = list->tail_->next_ = malloc(nnodebytes);
	}
	list->tail_->next_ = NULL;
	list->tail_->val_ = val;
}

#endif
