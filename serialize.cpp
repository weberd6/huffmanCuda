#include <fstream>
#include <sys/stat.h>

#include "main.h"

long getFileSize(std::string filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

void serialize_tree(Node* root, std::ofstream& ofs)
{
    if (root == NULL) {
        unsigned char all_ones = 255;
        ofs.write(reinterpret_cast<const char*>(&all_ones), sizeof(all_ones));
        ofs.write(reinterpret_cast<const char*>(&all_ones), sizeof(all_ones));
        ofs.write(reinterpret_cast<const char*>(&all_ones), sizeof(all_ones));
        return;
    }

    unsigned char i = root->symbol_index;
    ofs.write(reinterpret_cast<const char*>(&i), sizeof(i));

    serialize_tree(root->get_left_child(), ofs);
    serialize_tree(root->get_right_child(), ofs);
}

void deserialize_tree(Node* &root, std::ifstream& ifs)
{
    char ch;
    ifs.get(ch);

    if (-1 == ch) {
        ifs.get(ch);
        if (-1 == ch) {
            ifs.get(ch);
            if (-1 == ch) {
                return;
            }
            else {
                ifs.unget();
                ifs.unget();
            }
            
        } else {
            ifs.unget();
        }
    }

    root = new Node;
    root->symbol_index = ch;

    deserialize_tree(root->get_left_child(), ifs);
    deserialize_tree(root->get_right_child(), ifs);
}

void print_tree(Node* root, unsigned int &count)
{
    if (root == NULL)
        return;

    if (!root->get_left_child())
        std::cout << count++ << " " << root->symbol_index << std::endl;

    print_tree(root->get_left_child(), count);
    print_tree(root->get_right_child(), count);
}

void tree_to_array(NodeArray* nodes, unsigned int index, Node* root)
{
    if (NULL == root) {
        nodes[index].symbol_index = 803;
        return;
    }

    if (!root->get_left_child() && !root->get_right_child()) {
        nodes[index].left = -1;
        nodes[index].right = -1;
        nodes[index].symbol_index = root->symbol_index;
    } else {
        nodes[index].left = 2*index+1;
        nodes[index].right = 2*index+2;
    }

    tree_to_array(nodes, 2*index+1, root->get_left_child());
    tree_to_array(nodes, 2*index+2, root->get_right_child());
}

void print_array_tree(NodeArray* tree, unsigned int root, unsigned int &count)
{
    if (tree[root].symbol_index == 803)
        return;

    if (tree[root].left == -1) {
        std::cout << count++ << " " << (unsigned char)tree[root].symbol_index << std::endl;
    }

    print_array_tree(tree, 2*root+1, count);
    print_array_tree(tree, 2*root+2, count);
}








