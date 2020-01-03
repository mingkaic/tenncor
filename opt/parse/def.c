#include <stdio.h>

#include "opt/parse/def.h"

#ifdef PARSE_DEF_H

void kv_recursive_free (void* kv)
{
	if (NULL == kv)
	{
		return;
	}
	struct KeyVal* k = (struct KeyVal*) kv;
	numlist_clear(&k->val_);
	free(k);
}

struct Functor* new_functor (char* symbol, struct PtrList* attr)
{
	if (NULL == symbol)
	{
		return NULL;
	}
	size_t nbytes = sizeof(struct Functor);
	struct Functor* f = malloc(nbytes);
	memset(f, 0, nbytes);
	strncpy(f->name_, symbol, NSYMBOL);
	ptrlist_move(&f->attrs_, attr);
	if (NULL == attr)
	{
		f->attrs_.type_ = KV_PAIR;
	}
	return f;
}

void func_recursive_free (void* func)
{
	if (NULL == func)
	{
		return;
	}
	struct Functor* f = (struct Functor*) func;
	objs_clear(&f->attrs_);
	objs_clear(&f->args_);
	free(f);
}

struct TreeNode* new_numnode (double scalar)
{
	struct TreeNode* node = malloc(sizeof(struct TreeNode));
	node->type_ = SCALAR;
	node->val_.scalar_ = scalar;
	return node;
}

struct TreeNode* new_anynode (char* any)
{
	if (NULL == any)
	{
		return NULL;
	}
	struct TreeNode* node = malloc(sizeof(struct TreeNode));
	node->type_ = ANY;
	char* dst = node->val_.any_ = malloc(NSYMBOL);
	strncpy(dst, any, NSYMBOL);
	return node;
}

struct TreeNode* new_fncnode (struct Functor* f)
{
	if (NULL == f)
	{
		return NULL;
	}
	struct TreeNode* node = malloc(sizeof(struct TreeNode));
	node->type_ = FUNCTOR;
	node->val_.functor_ = f;
	return node;
}

void node_recursive_free (void* node)
{
	if (NULL == node)
	{
		return;
	}
	struct TreeNode* root = (struct TreeNode*) node;
	switch (root->type_)
	{
		case SCALAR:
			break;
		case ANY:
			free(root->val_.any_);
			break;
		case FUNCTOR:
			func_recursive_free(root->val_.functor_);
			break;
		default:
			fprintf(stderr, "freeing unknown node %d\n", root->type_);
	}
	free(root);
}

struct Conversion* new_conversion (
	struct Functor* matcher, struct TreeNode* target)
{
	if (NULL == matcher || NULL == target)
	{
		return NULL;
	}
	struct Conversion* cv = malloc(sizeof(struct Conversion));
	cv->matcher_ = matcher;
	cv->target_ = target;
	return cv;
}

void conversion_recursive_free (void* conv)
{
	if (NULL == conv)
	{
		return;
	}
	struct Conversion* c = (struct Conversion*) conv;
	func_recursive_free(c->matcher_);
	node_recursive_free(c->target_);
	free(c);
}

void objs_free (struct PtrList* objs)
{
	if (NULL == objs)
	{
		return;
	}
	void (*recursive_free)(void*) = NULL;
	switch (objs->type_)
	{
		case CONVERSION:
			recursive_free = conversion_recursive_free;
			break;
		case ARGUMENT:
			recursive_free = node_recursive_free;
			break;
		case KV_PAIR:
			recursive_free = kv_recursive_free;
			break;
		default:
			fprintf(stderr, "freeing unknown object %zu\n", objs->type_);
	}
	ptrlist_free(objs, recursive_free);
}

void objs_clear (struct PtrList* objs)
{
	void (*recursive_free)(void*) = NULL;
	switch (objs->type_)
	{
		case CONVERSION:
			recursive_free = &conversion_recursive_free;
			break;
		case ARGUMENT:
			recursive_free = &node_recursive_free;
			break;
		case KV_PAIR:
			recursive_free = &kv_recursive_free;
			break;
		default:
			fprintf(stderr, "freeing unknown object %zu\n", objs->type_);
	}
	ptrlist_clear(objs, recursive_free);
}

void cversions_free (struct PtrList* cversions)
{
	objs_free(cversions);
}

#endif
