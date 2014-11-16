#ifndef __NODE_H__
#define __NODE_H__

class Node {
	private:
		Node* left_child;
		Node* right_child;
		char* value;

	public:
		Node();
		Node(Node* left_child, Node* right_child, char* value);
		Node* get_left_child();
		Node* get_right_child();
		char* get_value();
		void set_left_child(Node* left_child);
		void set_right_child(Node* right_child);
		void set_value(char value[]);
		int frequency;
		int symbol_index;
};

#endif
