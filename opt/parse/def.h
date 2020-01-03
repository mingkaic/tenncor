///
/// def.h
/// opt/parse
///
/// Purpose:
/// Define c rule tree nodes and meta-data
///

#include <stdio.h>

#include "opt/parse/list.h"

#ifndef PARSE_DEF_H
#define PARSE_DEF_H

/// Length of symbols
#define NSYMBOL 32

#define FALSE 0
#define TRUE 1

/// Represent map key-value pair
struct KeyVal
{
	int val_scalar_;
	struct NumList val_;
	char key_[NSYMBOL];
};

/// Recursively free the key-value pair and all it's contents
void kv_recursive_free (void* kv);

/// Branching (TreeNode or Group) node
struct Functor
{
	struct PtrList args_;
	char name_[NSYMBOL];
	char variadic_[NSYMBOL];
	int commutative_;
	/// Argument attributes
	struct PtrList attrs_;
};

struct Functor* new_functor (char* symbol, struct PtrList* attr);

/// Recursively free the functor and all it's contents
void func_recursive_free (void* arg);

/// Rule tree node
struct TreeNode
{
	/// Rule tree node type
	enum SUBGRAPH_TYPE
	{
		SCALAR = 0, /// scalar_
		ANY, 		/// any_
		FUNCTOR, 	/// functor_
	} type_;
	/// Node instance according to type
	union
	{
		double scalar_;
		char* any_;
		struct Functor* functor_;
	} val_;
};

struct TreeNode* new_numnode (double scalar);

struct TreeNode* new_anynode (char* any);

struct TreeNode* new_fncnode (struct Functor* f);

/// Recursively free the subgraph and all of its descendants
void node_recursive_free (void* node);

/// Represent a conversion between two rule trees
struct Conversion
{
	/// Root of matching subgraph
	struct Functor* matcher_;

	/// Root of target subgraph
	struct TreeNode* target_;
};

struct Conversion* new_conversion (
	struct Functor* matcher, struct TreeNode* target);

/// Recursively free the conversion and its sub nodes
void conversion_recursive_free (void* conv);

/// Pointer enumeration to store in PtrList
enum PTR_TYPE
{
	CONVERSION = 0,
	ARGUMENT,
	KV_PAIR,
};

void objs_free (struct PtrList* objs);

void objs_clear (struct PtrList* objs);

#ifdef __cplusplus
extern "C" {
#endif

/// Recursively free all statements and their contents
extern void cversions_free (struct PtrList* cversions);

/// Return 0 if successfully populate statements from parsed string
extern int parse_str (struct PtrList** stmts, const char* str);

/// Return 0 if successfully populate statements from parsed file
extern int parse_file (struct PtrList** stmts, FILE* file);

#ifdef __cplusplus
}
#endif

#endif // PARSE_DEF_H
