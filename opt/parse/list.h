///
/// list.h
/// opt/parse
///
/// Purpose:
/// Define c lists needed to represent rule trees
///

#include <assert.h>
#include <stdlib.h>

#ifndef PARSE_LIST_H
#define PARSE_LIST_H

/// Node of a decimal linked list
struct NumNode
{
	/// Next node in list (null denoting end of list)
	struct NumNode* next_;

	/// Stored decimal value
	double val_;
};

/// Decimal linked list
struct NumList
{
	/// Head of the linked list
	struct NumNode* head_;

	/// Tail of the linked list (tail_.next_ should be null)
	struct NumNode* tail_;
};

/// Return a new decimal linked list
struct NumList* new_numlist (void);

/// Move all nodes from src to dst
void numlist_move (struct NumList* dst, struct NumList* src);

/// Free all nodes in list and reset list head and tail to null
void numlist_clear (struct NumList* list);

/// Clear list then free the list
void numlist_free (struct NumList* list);

/// Append val to linked list
void numlist_pushback (struct NumList* list, double val);


/// Node of a pointer linked list
struct PtrNode
{
	/// Next node in list (null denoting end of list)
	struct PtrNode* next_;

	/// Stored pointer value
	void* val_;
};

/// Pointer linked list
struct PtrList
{
	/// Head of the linked list
	struct PtrNode* head_;

	/// Tail of the linked list (tail_.next_ should be null)
	struct PtrNode* tail_;

	/// Enumerated type of stored pointer
	size_t type_;
};

/// Return a new pointer list of specified enumerated type
struct PtrList* new_ptrlist (size_t type);

/// Move all nodes from src to dst
void ptrlist_move (struct PtrList* dst, struct PtrList* src);

/// Frees all node in list and reset list head and tail to null
/// Before freeing for every pointer value x in list
/// call val_mgr(x) if val_mgr is not null
void ptrlist_clear (struct PtrList* list, void (*val_mgr)(void*));

/// Clear the list using val_mgr, then free the list
void ptrlist_free (struct PtrList* list, void (*val_mgr)(void*));

/// Append the pointer to the list
void ptrlist_pushback (struct PtrList* list, void* val);

#endif // PARSE_LIST_H
