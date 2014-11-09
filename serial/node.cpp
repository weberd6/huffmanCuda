#include "node.h"
#include <cstddef>

Node::Node() {
	this->left_child = NULL;
	this->right_child = NULL;
	this->value = -1;
}

Node::Node(Node* left_child, Node* right_child, int value) {
	this->left_child = left_child;
	this->right_child = right_child;
	this->value = value;
}

Node* Node::get_left_child() {
	return this->left_child;
}

Node* Node::get_right_child() {
	return this->right_child;
}

void Node::set_left_child(Node* left_child) {
	this->left_child = left_child;
}

void Node::set_right_child(Node* right_child) {
	this->right_child = right_child;
}

int Node::get_value() {
	return this->value;
}

void Node::set_value(int value) {
	this->value = value;
}