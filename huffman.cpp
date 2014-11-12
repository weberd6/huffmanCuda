
// GenerateCode
// Input:  root - the root of a 2-tree
// Output: Code[0:n-1] - array of binary strings, where Code[i] is the code for the symbol ai
void generate_code(Node root, int code[]) {
	if (!root.getLeftChild())
		code[root.symbolIndex()] = root.binaryCode();
	else
	{
		Node left = root.getLeftChild();
		Node right = root.getRightChild();
		left.binaryCode = root.binaryCode + '0';
		right.binaryCode = root.binaryCode + '1';
		generate_code(left, code);
		generate_code(right, code);
	}
}


// HuffmanCode
// Input:  Freq[0:n-1] - an array of non-negative frequencies, where Freq[i] == fi
// Output: Code[0:n-1] - an array of binary strings for Huffman code, where Code[i] is the binary string encoding symbol ai, i=0,...,n-1
void huffman_code(int a[], int freq[], int code[]) {
	for (int i=0; i<n; i++)	
	{
		Node p = new Node;
		allocate_node(p);
		p.symbolIndex = i;
		p.frequency = freq[i];
		leaf[i] = p;
	}

	create_priority_queue(leaf, q);

	for (int i=1; i<n; i++)
	{
		remove_priority_queue(Q, L);
		remove_priority_queue(Q, R);
		allocate_node(root);
		root.setLeftChild(L);
		root.setRightChild(R);
		root.frequency = (L.frequency) + (R.frequency);
		insert_priority_queue(Q, root);
	}

	root.binaryString = "";
	
	generate_code(root, code);
}
