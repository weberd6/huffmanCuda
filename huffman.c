#include "node.h"
#include "node.cpp"

void allocate_huffman_node(Node p);
void create_priority_queue(Node leaf[], int q[]);

// GenerateCode
// Input:  root - the root of a 2-tree
// Output: Code[0:n-1] - array of binary strings, where Code[i] is the code for the symbol ai
void generate_code(Node *root, int code[]) {
	if (!root->get_left_child())
		; // code[root.symbolIndex()] = root.binaryCode();
	else
	{
		Node *left = root->get_left_child();
		Node *right = root->get_right_child();
		left->set_value(root->get_value() + '0');
		right->set_value(root->get_value() + '1');
		generate_code(left, code);
		generate_code(right, code);
	}
}


// HuffmanCode
// Input:  Freq[0:n-1] - an array of non-negative frequencies, where Freq[i] == fi
// Output: Code[0:n-1] - an array of binary strings for Huffman code, where Code[i] is the binary string encoding symbol ai, i=0,...,n-1
void huffman_code(int a[], int freq[], int code[]) {
	int n = sizeof(&a);
	Node leaf[1];
	Node q[1];
	Node *l;
	Node *r;
	Node *root;

	// init leaf nodes
	for (int i=0; i<n; i++)
	{
		Node *p = new Node();
		p->symbol_index = i;
		p->frequency = freq[i];
		leaf[i] = *p;
	}


	// std::priority_queue<int, std::vector<int>, std::greater<int> > q;
	// create_priority_queue(leaf, q);



	for (int i=1; i<n; i++)
	{
		// remove smallest and second smallest frequencies from the queue

		// remove_priority_queue(q, l);
		// remove_priority_queue(q, r);

		// create a new root node;
		root = new Node();

		root->set_left_child(l);
		root->set_right_child(r);
		root->frequency = (l->frequency) + (r->frequency);
		// insert_priority_queue(q, root);
	}

	root->set_value(NULL);

	generate_code(root, code);
}
