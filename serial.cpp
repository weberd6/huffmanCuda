#include <queue>
#include <vector>
#include <stdlib.h>
#include "node.h"
#include "main.h"

const int SIZE = 6;

unsigned int mask[32] = {0, 1, 3, 7, 15, 31, 63, 127,
            255, 511, 1023, 2047, 4095, 8191, 16383, 32767,
            65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 8388607,
            16777215, 33554431, 67108863, 134217727, 268435455, 536870911, 1073741823, 2147483647};


// GenerateCode - generates a binary prefix code for a 2-tree
// Input:  root - the root of a 2-tree
// Output: Code[0:n-1] - array of binary strings, where Code[i] is the code for the symbol ai
void generate_code(Node *root, unsigned int code[], unsigned int length[]) {
    if (!root->get_left_child())
    {
        code[root->symbol_index] = root->get_value();
        length[root->symbol_index] = root->length;
    }
    else
    {
        Node *left = root->get_left_child();
        Node *right = root->get_right_child();
        left->set_value(root->get_value());
        right->set_value(root->get_value() | (mask[root->length]+1));
        left->length = root->length + 1;
        right->length = root->length + 1;
        generate_code(left, code, length);
        generate_code(right, code, length);
    }
}


// HuffmanCode
// Input:  a[], representing an alphabet, where a[i] == ai,
//         Freq[0:n-1] - an array of non-negative frequencies, where Freq[i] == fi
// Output: Code[0:n-1] - an array of binary strings for Huffman code, where Code[i] is the binary string encoding symbol ai, i=0,...,n-1
void huffman_code(int a[], int freq[], unsigned int code[]) {
  int n = SIZE;  // n is the alphabet size
  std::priority_queue<Node*, std::vector<Node*>, NodeGreater> q;

  // init leaf nodes
  for (int i=0; i<n; i++)
    {
      Node *p = new Node();
      p->symbol_index = i;
      p->frequency = freq[i];
      q.push(p);
    }


    while (q.size() > 1)
      {
        // remove smallest and second smallest frequencies from the queue
        Node *l = q.top();
        q.pop();

        Node *r = q.top();
        q.pop();

        // create a new subtree with the smallest nodes
        Node *subtree = new Node();
        subtree->set_left_child(l);
        subtree->set_right_child(r);

        // the new root's frequency is the sum of the children's frequencies
        subtree->frequency = (l->frequency) + (r->frequency);

        // insert the subtree into the heap
        q.push(subtree);
      }

      Node *root = q.top();

      generate_code(root, code, b);
}
