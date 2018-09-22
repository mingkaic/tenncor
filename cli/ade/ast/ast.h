#ifdef __cplusplus
extern "C"
{
#endif

enum FUNCODE
{
	ABS = 0,
	NEG,
	NOT,
	SIN,
	COS,
	TAN,
	EXP,
	LOG,
	SQRT,
	ROUND,
	FLIP,

	POW,
	ADD,
	SUB,
	MUL,
	DIV,
	EQ,
	NE,
	LT,
	GT,

	BINO,
	UNIF,
	NORM,

	NELEMS,
	NDIMS,

	ARGMAX,
	RMAX,
	RSUM,

	MATMUL,

	PERMUTE,
	EXTEND,
	RESHAPE,
};


// shape
struct ShapeHolder;
// caller must free instance using free_shape
struct ShapeHolder* make (int dim);
void free_shape (struct ShapeHolder* shape);
void append (struct ShapeHolder* dest, int dim);


// ast
struct ASTNode;
// caller must free instance using free_ast
struct ASTNode* empty_node (void);
struct ASTNode* to_node (const struct ShapeHolder* shape);
void free_ast (struct ASTNode* node);
void save_ast (char key[32], struct ASTNode* value);

struct ASTNode* load_ast (char key[32]);


// operators
struct ASTNode* unary (struct ASTNode* node, enum FUNCODE code);
struct ASTNode* binary (struct ASTNode* node,
	struct ASTNode* node2, enum FUNCODE code);
struct ASTNode* shapeop (struct ASTNode* node,
	struct ShapeHolder* shape, enum FUNCODE code);
struct ASTNode* grad (struct ASTNode* node, struct ASTNode* wrt);


// display
void use_mode (char mode[32]);
void show_shape (struct ASTNode* node);
void show_eq (struct ASTNode* node);

#ifdef __cplusplus
}
#endif
