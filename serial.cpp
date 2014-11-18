#include <queue>
#include <vector>
#include <stdlib.h>
#include "node.h"

// GenerateCode
// Input:  root - the root of a 2-tree
// Output: Code[0:n-1] - array of binary strings, where Code[i] is the code for the symbol ai
void generate_code(Node *root, int code[]) {
	if (!root->get_left_child())
		code[root->symbol_index] = atoi(root->get_value());
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
	std::priority_queue<Node*, std::vector<Node*>, NodeGreater> q;
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

	for (int i=1; i<n; i++)
	{
		// remove smallest and second smallest frequencies from the queue
		Node *l = q.top();
		q.pop();

		Node *r = q.top();
		q.pop();

		// create a new subtree
		root = new Node();
		root->set_left_child(l);
		root->set_right_child(r);

		// the frequency of the root is the sum of the children's frequencies
		root->frequency = (l->frequency) + (r->frequency);

		// insert the subtree into the heap
		q.push(root);
	}

	root->set_value(NULL);

	generate_code(root, code);
}
