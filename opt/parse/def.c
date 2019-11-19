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

void arg_recursive_free (void* arg)
{
	if (NULL == arg)
	{
		return;
	}
	struct Arg* a = (struct Arg*) arg;
	node_recursive_free(a->node_);
	objs_clear(&a->attrs_);
	free(a);
}

void func_recursive_free (void* func)
{
	if (NULL == func)
	{
		return;
	}
	struct Functor* f = (struct Functor*) func;
	objs_clear(&f->args_);
	free(f);
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
			recursive_free = arg_recursive_free;
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
			recursive_free = &arg_recursive_free;
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
