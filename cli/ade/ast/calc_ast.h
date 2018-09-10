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

struct ShapeHolder;

void use_mode (char mode[32]);

// caller must free instance using free_shape
struct ShapeHolder* make (int dim);
void free_shape (struct ShapeHolder* shape);
void append (struct ShapeHolder* dest, int dim);

struct ASTNode;

// caller must free instance using free_ast
struct ASTNode* emptyNode (void);
struct ASTNode* toNode (const struct ShapeHolder* shape);
void free_ast (struct ASTNode* node);
void saveAST (char key[32], struct ASTNode* value);

struct ASTNode* loadAST (char key[32]);

// operators
struct ASTNode* unary (struct ASTNode* node, enum FUNCODE code);
struct ASTNode* binary (struct ASTNode* node,
	struct ASTNode* node2, enum FUNCODE code);
struct ASTNode* shapeop (struct ASTNode* node,
	struct ShapeHolder* shape, enum FUNCODE code);
struct ASTNode* grad (struct ASTNode* node, struct ASTNode* wrt);

// display
// write out node's shape
void show_shape (struct ASTNode* node);
// print out subgraph node
void show_eq (struct ASTNode* node);

#ifdef __cplusplus
}
#endif
